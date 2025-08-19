import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from openai import OpenAI
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
# NEW: Import the library for creating sparse vectors for keyword search
from pinecone_text.sparse import BM25Encoder

# --- 1. Load Environment Variables and Initialize Clients ---
load_dotenv()

# Initialize OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found.")
    openai_client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized.")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    exit()

# Initialize Pinecone client
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found.")
    pc = Pinecone(api_key=pinecone_api_key)
    
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    if not INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME not found.")
        
    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index '{INDEX_NAME}'.")
except Exception as e:
    print(f"Failed to initialize Pinecone client or connect to index: {e}")
    exit()

# NEW: Initialize the BM25Encoder for creating sparse vectors
# This encoder is used for the keyword search part of hybrid search.
# It does not need to be "fit" on data for this general use case.
bm25_encoder = BM25Encoder.default()
print("BM25 encoder for sparse vectors initialized.")


# --- 2. FastAPI App Setup ---
app = FastAPI(
    title="Blue Banner Bot API",
    description="An API to ask questions about robotics competition documentation.",
    version="1.0.0"
)

# Add CORS Middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = Field(default_factory=list)

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-5-mini"


# --- 3. The Core RAG Logic in an API Endpoint ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    This endpoint receives a question and chat history, retrieves context,
    and uses GPT-4o to generate a conversational answer.
    """
    try:
        # --- NEW: Hybrid Search Logic ---
        # Step 1: Create the DENSE vector for semantic search
        print(f"Creating dense vector for question: '{request.question}'")
        dense_vector = openai_client.embeddings.create(
            input=[request.question],
            model=EMBEDDING_MODEL
        ).data[0].embedding

        # Step 2: Create the SPARSE vector for keyword search
        print(f"Creating sparse vector for question: '{request.question}'")
        sparse_vector = bm25_encoder.encode_queries(request.question)

        # Step 3: Query Pinecone using both vectors for hybrid search
        print("Querying Pinecone with hybrid search...")
        query_results = index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=5, # Retrieve a mix of the best 5 results from both search types
            include_metadata=True
        )
        
        context_chunks = [match['metadata']['text'] for match in query_results['matches']]
        context_string = "\n---\n".join(context_chunks)
        
        if not context_string:
            print("No relevant context found in Pinecone.")
            context_string = "No relevant documents found."

        # Step 4: Combine history and new question for the prompt (Memory)
        system_prompt = """
        You are a helpful robotics competition technical assistant called Blue Banner Bot. 
        Answer the user's question based on the provided chat history and the retrieved context documents.
        Be concise and clear in your explanation. If the context doesn't contain the answer,
        say that you couldn't find the information in the provided documents.
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "system", "content": f"Retrieved Context:\n{context_string}"})
        messages.extend(request.history)
        messages.append({"role": "user", "content": request.question})

        # Step 5: Send the complete conversation to GPT-4o
        print("Sending request to GPT-4o for final answer...")
        completion_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        
        final_answer = completion_response.choices[0].message.content
        print(f"Received answer: {final_answer}")
        
        return {"answer": final_answer}

    except Exception as e:
        print(f"An error occurred in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Blue Banner Bot API"}
