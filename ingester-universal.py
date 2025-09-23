import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# --- 1. Configuration: Add new directories to this list ---
DIRECTORIES_TO_INGEST = [
    {"path": "2025_game_manual", "content_type": "text"},
]

# --- 2. Load Environment Variables and Initialize Clients ---
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

def chunk_text(text, content_type, chunk_size=1000, chunk_overlap=200):
    """
    Splits text using the appropriate chunker based on content type.
    """
    if content_type == 'code':
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    else: # Default to a general text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    return text_splitter.split_text(text)

def get_embedding(text):
    try:
        response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"  - Error creating embedding: {e}")
        return None

# --- 3. Main Ingestion Logic ---
def main():
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating it now...")
        pc.create_index(
            name=INDEX_NAME, dimension=EMBEDDING_DIMENSIONS, metric="dotproduct",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(5)
    
    index = pc.Index(INDEX_NAME)
    print(f"Successfully connected to index '{INDEX_NAME}'.")

    # --- NEW: Rate Limiting Variables ---
    request_counter = 0
    start_time = time.time()
    # ------------------------------------

    for directory_info in DIRECTORIES_TO_INGEST:
        input_directory = directory_info["path"]
        content_type = directory_info["content_type"]
        
        print(f"\n{'='*50}\nProcessing directory: {input_directory} (Type: {content_type})\n{'='*50}")

        if not os.path.exists(input_directory):
            print(f"Warning: Directory '{input_directory}' not found. Skipping.")
            continue

        for filename in os.listdir(input_directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(input_directory, filename)
                print(f"\n--- Processing file: {filename} ---")
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                
                if not file_text.strip():
                    print("  - File is empty, skipping.")
                    continue

                chunks = chunk_text(file_text, content_type)
                print(f"  - Created {len(chunks)} chunks using '{content_type}' splitter.")

                batch_size = 100
                vectors_to_upsert = []
                for i, chunk in enumerate(chunks):
                    # --- MODIFIED: Rate Limiting Logic ---
                    if request_counter >= 59: # Check if we are near the limit
                        elapsed_time = time.time() - start_time
                        if elapsed_time < 60:
                            wait_time = 60 - elapsed_time
                            print(f"  - Rate limit approaching. Pausing for {wait_time:.2f} seconds...")
                            time.sleep(wait_time)
                        # Reset the counter and timer for the next minute
                        request_counter = 0
                        start_time = time.time()
                    # -------------------------------------

                    embedding = get_embedding(chunk)
                    request_counter += 1 # Increment after each request
                    
                    if embedding:
                        vector_id = f"{input_directory}-{filename}-{i}"
                        vectors_to_upsert.append({
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {"text": chunk, "source": filename}
                        })
                    
                    if len(vectors_to_upsert) >= batch_size or (i == len(chunks) - 1 and vectors_to_upsert):
                        print(f"  - Upserting batch of {len(vectors_to_upsert)} vectors to Pinecone...")
                        index.upsert(vectors=vectors_to_upsert)
                        print("  - Batch upsert complete.")
                        vectors_to_upsert = []

    print("\n--- All Ingestion Complete ---")
    print(f"Final index stats: {index.describe_index_stats()}")

if __name__ == "__main__":
    main()