import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# --- 1. Load Environment Variables and Initialize Clients ---
# This section is similar to your ingestion script.
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
    title="FRC AI Assistant API",
    description="An API to ask questions about FRC documentation.",
    version="1.0.0"
)

# This defines the structure of the request body for the /ask endpoint
class QueryRequest(BaseModel):
    question: str

# The OpenAI model to use for creating embeddings for the query
EMBEDDING_MODEL = "text-embedding-3-small"
# The OpenAI model to use for generating the final answer
GPT_MODEL = "gpt-4o"


# --- 3. The Core RAG Logic in an API Endpoint ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    This endpoint receives a question, retrieves relevant context from Pinecone,
    and uses GPT-4o to generate an answer.
    """
    try:
        # Step 1: Create an embedding for the user's question
        print(f"Creating embedding for question: '{request.question}'")
        query_embedding = openai_client.embeddings.create(
            input=[request.question],
            model=EMBEDDING_MODEL
        ).data[0].embedding

        # Step 2: Query Pinecone to find the most relevant text chunks
        print("Querying Pinecone for relevant context...")
        query_results = index.query(
            vector=query_embedding,
            top_k=5,  # Retrieve the top 5 most relevant chunks
            include_metadata=True
        )
        
        # Step 3: Combine the retrieved chunks into a single context string
        context_chunks = [match['metadata']['text'] for match in query_results['matches']]
        context_string = "\n---\n".join(context_chunks)
        
        if not context_string:
            print("No relevant context found in Pinecone.")
            return {"answer": "I'm sorry, I couldn't find any relevant information in my documents to answer that question."}

        # Step 4: Build the prompt and send it to GPT-4o
        print("Sending request to GPT-4o for final answer...")
        system_prompt = """
        You are a helpful FRC (FIRST Robotics Competition) technical assistant. 
        Answer the user's question based ONLY on the context provided below.
        Be concise and clear in your explanation. If the context doesn't contain the answer,
        say that you couldn't find the information in the provided documents.
        """
        
        completion_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"""
                    CONTEXT:
                    {context_string}
                    
                    QUESTION:
                    {request.question}
                    """
                }
            ]
        )
        
        final_answer = completion_response.choices[0].message.content
        print(f"Received answer: {final_answer}")
        
        return {"answer": final_answer}

    except Exception as e:
        print(f"An error occurred in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# A simple root endpoint to confirm the server is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the FRC AI Assistant API"}

