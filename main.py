import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from openai import OpenAI
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
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

# Initialize the BM25Encoder for creating sparse vectors
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

# NEW: Pydantic model for the summary request
class SummaryRequest(BaseModel):
    history: List[Dict[str, str]]

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-5-mini"


# --- 3. New Endpoint for Summarization ---
@app.post("/summarize")
async def summarize_history(request: SummaryRequest):
    """
    Summarizes the chat history using a language model.
    """
    try:
        print("Summarizing chat history...")
        
        # Prepare the conversation for the summarization prompt
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.history])
        
        summary_prompt = f"""
        Please provide a concise summary of the following conversation history.
        Do not add any conversational text like "Here is a summary".
        The summary should be objective and capture the key topics discussed.
        
        ---
        Conversation:
        {conversation_text}
        """
        
        # Call the LLM to get the summary
        summary_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        summary = summary_response.choices[0].message.content.strip()
        print(f"Summary created: {summary}")
        
        return {"summary": summary}
        
    except Exception as e:
        print(f"An error occurred in /summarize endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# --- 4. The Core RAG Logic with Summary Integration ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    This endpoint receives a question and chat history, retrieves context,
    and uses GPT-4o to generate a conversational answer.
    """
    try:
        # Check if the history is too long and needs to be summarized.
        # This is a simple threshold; you can adjust it.
        # For example, summarize every 10-15 messages.
        if len(request.history) > 10:
            print("Chat history is long, requesting a summary...")
            summary_response = await summarize_history(SummaryRequest(history=request.history))
            summary = summary_response["summary"]
            
            # Replace the long history with a single summary message.
            # We keep the last few messages for immediate context.
            # This is a hybrid approach, but better than a full history.
            summary_message = {"role": "system", "content": f"Summary of conversation so far: {summary}"}
            last_few_messages = request.history[-4:]
            request.history = [summary_message] + last_few_messages
            print("History has been summarized and updated.")

        # --- Hybrid Search Logic ---
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
            top_k=5, 
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