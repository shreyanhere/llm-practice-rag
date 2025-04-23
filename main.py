import os
import asyncio
import logging
import uvicorn
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- LangChain and Related Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as PydanticBaseModel, Field

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

from typing import Literal, List, Dict, Any
from typing_extensions import TypedDict
from langchain.schema import Document
import cassio
from langgraph.graph import END, StateGraph, START

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment variables from .env file ---
load_dotenv()

# --- Initialization Section ---
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure required variables are present.
for var_name, var_value in {
    "ASTRA_DB_APPLICATION_TOKEN": ASTRA_DB_APPLICATION_TOKEN,
    "ASTRA_DB_ID": ASTRA_DB_ID,
}.items():
    if not var_value:
        raise ValueError(f"Missing environment variable: {var_name}")


cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# --- Index Building Section ---
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://skphd.medium.com/top-25-langchain-interview-questions-and-answers-d84fb23576c8/",
]

def load_and_split_documents(url_list: List[str]) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in url_list]
    docs_flat = [doc for sublist in docs for doc in sublist]
    logger.info("Loaded %i documents from URLs.", len(docs_flat))
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,  
        chunk_overlap=150,  
    )
    doc_splits = text_splitter.split_documents(docs_flat)
    logger.info("Document chunks created: %i", len(doc_splits))
    return doc_splits

doc_splits = load_and_split_documents(urls)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

astra_vector_store.add_documents(doc_splits)
logger.info("Inserted %i document chunks into the vectorstore.", len(doc_splits))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

retriever = astra_vector_store.as_retriever(search_kwargs={"k": 5})

# --- Router Setup ---
class RouteQuery(PydanticBaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Route the query either to the vectorstore or a wiki search."
    )

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = (
    "You are an expert at routing a user question to a vectorstore or Wikipedia. "
    "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks. "
    "Use the vectorstore for questions on these topics; otherwise, use Wikipedia search."
)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# --- External Tools ---
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# --- Graph Construction ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    refined_answer: str
    context: str

def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("--- RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    logger.debug("Documents returned from retriever: %s", documents)
    return {"documents": documents, "question": question}

def wiki_search(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("--- WIKIPEDIA SEARCH ---")
    question = state["question"]
    try:
        result = wiki.invoke({"query": question})
        logger.debug("Raw Wikipedia response: %s", result)
        wiki_doc = Document(
            page_content=str(result),
            metadata={"source": "wikipedia"}
        )
        return {"documents": [wiki_doc], "question": question}
    except Exception as e:
        logger.error("Wikipedia error: %s", str(e))
        return {"documents": [], "question": question}

def refine(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("--- REFINE ---")
    documents = state.get("documents", [])
    question = state.get("question")
    
    # Create context by concatenating document contents.
    context = "\n\n".join([doc.page_content for doc in documents])
    state["context"] = context  # Save context in state.
    prompt = (
        f"Using the following context:\n\n{context}\n\n"
        f"And the original question: '{question}', produce a refined and coherent answer."
    )
    
    refined_response = llm.invoke(prompt)
    if isinstance(refined_response, dict):
        raw_answer = refined_response.get("result", "")
    else:
        raw_answer = str(refined_response)
    
    # Improved extraction logic for answer.
    if "content='" in raw_answer and "' additional_kwargs" in raw_answer:
        refined_answer = raw_answer.split("content='", 1)[1].split("' additional_kwargs", 1)[0]
    elif 'content="' in raw_answer and '" additional_kwargs' in raw_answer:
        refined_answer = raw_answer.split('content="', 1)[1].split('" additional_kwargs', 1)[0]
    else:
        refined_answer = raw_answer

    logger.debug("Refined Answer: %s", refined_answer)
    state["refined_answer"] = refined_answer
    return state

def route_question(state: Dict[str, Any]) -> str:
    logger.info("--- ROUTE QUESTION ---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if getattr(source, "datasource", None) == "wiki_search":
        logger.info("--- Routed to Wiki Search ---")
        return "wiki_search"
    logger.info("--- Defaulted to Vectorstore Retrieval ---")
    return "vectorstore"

workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("refine", refine)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve", 
    },
)
workflow.add_edge("retrieve", "refine")
workflow.add_edge("wiki_search", "refine")
workflow.add_edge("refine", END)
graph_app = workflow.compile()

# --- FastAPI Setup ---
app = FastAPI(title="Multi AI Agents RAG API with FastAPI")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    result: str

def extract_result(final_state: Dict[str, Any]) -> Dict[str, str]:
    try:
        # Get refined answer if available; otherwise, get the first document's content.
        if final_state.get("refined_answer"):
            answer = final_state["refined_answer"]
        else:
            docs = final_state.get("documents", [])
            answer = docs[0].page_content if docs else "No results found"
        # Also include context if available.
        context = final_state.get("context", "")
        return {"result": answer, "context": context}
    except Exception as e:
        logger.error("Extraction error: %s", str(e))
        return {"result": "Error processing results", "context": ""}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Empty query")
        
        logger.info("Processing query: '%s'", question)
        inputs = {"question": question}
        state: Dict[str, Any] = {}
        
        for output in graph_app.stream(inputs):
            logger.debug("Node output: %s", output)
            for node_data in output.values():
                state.update(node_data)
        
        logger.debug("Accumulated state after stream: %s", state)
        
        if not state.get("documents") and not state.get("refined_answer"):
            raise HTTPException(status_code=500, detail="No final state generated")
        
        final = extract_result(state)
        logger.info("Final result (first 200 chars): %s", final["result"][:200])
        return QueryResponse(result=final["result"])
        
    except Exception as e:
        logger.critical("CRITICAL ERROR: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
