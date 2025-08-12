import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from openai import OpenAI
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware

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

# --- NEW: Updated Request Body to include chat history ---
class QueryRequest(BaseModel):
    question: str
    # The history is a list of dictionaries, e.g., [{"role": "user", "content": "Hello"}]
    history: List[Dict[str, str]] = Field(default_factory=list)

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o"


# --- 3. The Core RAG Logic in an API Endpoint ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    This endpoint receives a question and chat history, retrieves context,
    and uses GPT-4o to generate a conversational answer.
    """
    try:
        # Step 1: Create an embedding for the user's question
        print(f"Creating embedding for question: '{request.question}'")
        query_embedding = openai_client.embeddings.create(
            input=[request.question],
            model=EMBEDDING_MODEL
        ).data[0].embedding

        # Step 2: Query Pinecone for relevant context
        print("Querying Pinecone for relevant context...")
        query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        context_chunks = [match['metadata']['text'] for match in query_results['matches']]
        context_string = "\n---\n".join(context_chunks)
        
        if not context_string:
            print("No relevant context found in Pinecone.")
            context_string = "No relevant documents found."

        # --- NEW: Combine history and new question for the prompt ---
        system_prompt = """
        You are a helpful robotics competition technical assistant called Blue Banner Bot. 
        Answer the user's question based on the provided chat history and the retrieved context documents.
        Be concise and clear in your explanation. If the context doesn't contain the answer,
        say that you couldn't find the information in the provided documents.
        """
        
        # The messages list starts with the system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add the retrieved context as a system message for the AI to reference
        messages.append({"role": "system", "content": f"Retrieved Context:\n{context_string}"})
        
        # Add the past conversation history
        messages.extend(request.history)
        
        # Add the user's current question
        messages.append({"role": "user", "content": request.question})

        # Step 4: Send the complete conversation to GPT-4o
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
