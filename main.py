import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from openai import OpenAI
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware
from pinecone_text.sparse import BM25Encoder
from datetime import datetime

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

class SummaryRequest(BaseModel):
    history: List[Dict[str, str]]

class FeedbackRequest(BaseModel):
    message: str

# Define models for specific tasks
EMBEDDING_MODEL = "text-embedding-3-small"
MAIN_ANSWER_MODEL = "gpt-5-mini" # For main RAG answer and summarization
GUARD_MODEL = "gpt-4o-mini"     # For fast, cheap topic classification

# --- Guard Statement Function ---
async def is_question_on_topic(question: str) -> bool:
    """
    Uses an LLM call to classify if the question is related to robotics competitions.
    """
    print(f"Checking if question is on-topic with {GUARD_MODEL}: '{question}'")
    
    classifier_prompt = f"""
    You are a topic classification model. Your sole purpose is to determine if a user's question is related to robotics competitions (like FRC, VEX, FTC), rules, robot design, game strategy, or technical specifications.

    Respond with only the word 'YES' or 'NO'. Do not provide any other text or explanation.

    Here are some examples:
    - User question: "How many falcons can I have on my robot?" -> Your response: YES
    - User question: "What is the capital of France?" -> Your response: NO
    - User question: "What are the rules about bumper construction?" -> Your response: YES
    - User question: "Write me a story about a dragon." -> Your response: NO

    ---
    User question: "{question}"
    Your response:
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": classifier_prompt}],
            max_tokens=5,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip().upper()
        print(f"On-topic classification result: {result}")
        return result == "YES"
        
    except Exception as e:
        print(f"Error during on-topic check: {e}")
        # Fail open: If the check fails, assume it's on-topic to not block valid questions.
        return True

# --- 3. Endpoint for Feedback with Structured Logging ---
@app.post("/feedback")
async def receive_feedback(request: FeedbackRequest):
    """
    Receives feedback from the user and logs it to Cloud Logging.
    """
    try:
        feedback_entry = {
            "event_type": "feedback_submitted",
            "timestamp": datetime.now().isoformat(),
            "message": request.message,
            "message_length": len(request.message)
        }
        
        print(json.dumps(feedback_entry))
        
        return {"status": "success", "message": "Feedback received. Thank you!"}
        
    except Exception as e:
        error_log = {
            "event_type": "error_occurred",
            "endpoint": "/feedback",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e)
        }
        print(json.dumps(error_log), file=os.sys.stderr)
        raise HTTPException(status_code=500, detail="Failed to process feedback.")

# --- 4. Endpoint for Summarization ---
@app.post("/summarize")
async def summarize_history(request: SummaryRequest):
    """
    Summarizes the chat history using a language model.
    """
    try:
        print(f"Summarizing chat history with {MAIN_ANSWER_MODEL}...")
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.history])
        
        summary_prompt = f"""
        Please provide a concise summary of the following conversation history.
        Do not add any conversational text like "Here is a summary".
        The summary should be objective and capture the key topics discussed.
        ---
        Conversation:
        {conversation_text}
        """
        
        summary_response = openai_client.chat.completions.create(
            model=MAIN_ANSWER_MODEL,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        summary = summary_response.choices[0].message.content.strip()
        print(f"Summary created: {summary}")
        
        return {"summary": summary}
        
    except Exception as e:
        error_log = {
            "event_type": "error_occurred",
            "endpoint": "/summarize",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e)
        }
        print(json.dumps(error_log), file=os.sys.stderr)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# --- 5. The Core RAG Logic with Structured Logging ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    This endpoint receives a question and chat history, retrieves context,
    and uses a model to generate a conversational answer.
    """
    try:
        # Step 0: Check if the question is on-topic before doing anything else.
        if not await is_question_on_topic(request.question):
            print("Question is off-topic. Returning canned response.")
            return {"answer": "I am the Blue Banner Bot, a robotics competition assistant. I can only answer questions related to robotics rules, manuals, and technical specifications. How can I help you with that?"}

        # Summarize if history is long
        if len(request.history) > 10:
            print("Chat history is long, requesting a summary...")
            summary_response = await summarize_history(SummaryRequest(history=request.history))
            summary = summary_response["summary"]
            
            summary_message = {"role": "system", "content": f"Summary of conversation so far: {summary}"}
            last_few_messages = request.history[-4:]
            request.history = [summary_message] + last_few_messages
            print("History has been summarized and updated.")

        # Step 1: Create the DENSE vector
        print(f"Creating dense vector for question: '{request.question}'")
        dense_vector = openai_client.embeddings.create(
            input=[request.question],
            model=EMBEDDING_MODEL
        ).data[0].embedding

        # Step 2: Create the SPARSE vector
        print(f"Creating sparse vector for question: '{request.question}'")
        sparse_vector = bm25_encoder.encode_queries(request.question)

        # Step 3: Query Pinecone
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

        # Step 4: Combine history and new question for the prompt
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

        # Step 5: Send the complete conversation to the main model
        print(f"Sending request to {MAIN_ANSWER_MODEL} for final answer...")
        completion_response = openai_client.chat.completions.create(
            model=MAIN_ANSWER_MODEL,
            messages=messages
        )
        
        final_answer = completion_response.choices[0].message.content
        print(f"Received answer: {final_answer}")
        
        # Structured Logging
        question_log_entry = {
            "event_type": "question_asked",
            "timestamp": datetime.now().isoformat(),
            "question_length": len(request.question),
            "conversation_length": len(request.history) + 1,
            "on_topic": True
        }
        print(json.dumps(question_log_entry))
        
        return {"answer": final_answer}

    except Exception as e:
        error_log = {
            "event_type": "error_occurred",
            "endpoint": "/ask",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e)
        }
        print(json.dumps(error_log), file=os.sys.stderr)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Blue Banner Bot API"}