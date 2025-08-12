import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Load Environment Variables and Initialize Clients ---
# Explicitly find the path to the .env file and load it.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)


# Initialize the OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
    openai_client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    exit()

# Initialize the Pinecone client
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    pc = Pinecone(api_key=pinecone_api_key)
    print("Pinecone client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Pinecone client: {e}")
    exit()

# --- 2. Configuration and Helper Functions ---
# Directory containing the scraped text files
# UPDATED: Changed this to your new javadoc folder
INPUT_DIRECTORY = "scraped_data_javadoc"
# The name of the Pinecone index we want to upload to
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# The OpenAI model to use for creating embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
# The dimensionality of the embedding model's output
EMBEDDING_DIMENSIONS = 1536

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    (Copied from the previous script)
    Splits a given text into smaller chunks.
    """
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

def get_embedding(text):
    """
    Generates an embedding for a given text using OpenAI's API.
    """
    try:
        response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"  - Error creating embedding: {e}")
        return None

# --- 3. Main Ingestion Logic ---
def main():
    """
    Main function to process files, create embeddings, and upload to Pinecone.
    """
    if not os.path.exists(INPUT_DIRECTORY):
        print(f"Error: Directory '{INPUT_DIRECTORY}' not found.")
        return

    # Check if the target index exists, and create it if it doesn't.
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating it now...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine", # Cosine is standard for text similarity
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1' # You can change this to your preferred region
            )
        )
        # Wait for the index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(5)
        print(f"Index '{INDEX_NAME}' created and ready.")

    # Connect to the index
    index = pc.Index(INDEX_NAME)
    print(f"Successfully connected to index '{INDEX_NAME}'.")
    print(f"Index stats: {index.describe_index_stats()}")

    # Process each file in the input directory
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_DIRECTORY, filename)
            print(f"\n--- Processing file: {filename} ---")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                
                if not file_text.strip():
                    print("  - File is empty, skipping.")
                    continue

                # Chunk the text from the file
                chunks = chunk_text(file_text)
                print(f"  - Created {len(chunks)} chunks.")

                vectors_to_upsert = []
                chunk_id_counter = 0
                for i, chunk in enumerate(chunks):
                    print(f"  - Creating embedding for chunk {i+1}/{len(chunks)}...")
                    embedding = get_embedding(chunk)
                    
                    if embedding:
                        # Create a unique ID for each chunk
                        vector_id = f"{filename}-{chunk_id_counter}"
                        chunk_id_counter += 1
                        
                        # We store the original text in the metadata
                        vectors_to_upsert.append({
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {"text": chunk}
                        })
                
                # Upsert in batches to be more efficient
                if vectors_to_upsert:
                    print(f"  - Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
                    index.upsert(vectors=vectors_to_upsert)
                    print("  - Batch upsert complete.")

            except Exception as e:
                print(f"  - An error occurred while processing {filename}: {e}")

    print("\n--- Ingestion Complete ---")
    print(f"Final index stats: {index.describe_index_stats()}")


# --- Main execution block ---
if __name__ == "__main__":
    # Before running, make sure you have the required libraries:
    # pip install python-dotenv openai pinecone langchain-text-splitters
    main()
