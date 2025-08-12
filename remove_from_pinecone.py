import os
from dotenv import load_dotenv
from pinecone import Pinecone

# --- 1. Load Environment Variables and Initialize Client ---
load_dotenv()

try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    
    if not pinecone_api_key or not INDEX_NAME:
        raise ValueError("PINECONE_API_KEY or PINECONE_INDEX_NAME not found in .env file.")
        
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    print(f"Successfully connected to Pinecone index '{INDEX_NAME}'.")

except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    exit()

# --- 2. Configuration ---
# The directory that contains the files you want to delete entries for.
# This should match the folder you accidentally ingested with the wrong chunker.
TARGET_DIRECTORY = "scraped_data_javadoc"


# --- 3. Main Deletion Logic ---
def main():
    """
    Finds and deletes all vectors associated with files from a specific directory.
    """
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"Error: Directory '{TARGET_DIRECTORY}' not found. No files to process for deletion.")
        return

    print(f"Preparing to delete all vectors derived from the '{TARGET_DIRECTORY}' folder.")
    
    # Get the list of filenames from the target directory
    target_filenames = [f for f in os.listdir(TARGET_DIRECTORY) if f.endswith(".txt")]
    
    if not target_filenames:
        print("No .txt files found in the target directory. Nothing to delete.")
        return

    print(f"Found {len(target_filenames)} files to target for deletion.")

    # Pinecone's list() paginates, so we need to loop to get all IDs.
    all_vector_ids = []
    print("Fetching all vector IDs from the index...")
    for ids in index.list(prefix=''):
        all_vector_ids.extend(ids)
    print(f"Total vectors in index: {len(all_vector_ids)}")

    # Find all vector IDs that start with one of our target filenames
    ids_to_delete = []
    for vector_id in all_vector_ids:
        for filename in target_filenames:
            if vector_id.startswith(filename):
                ids_to_delete.append(vector_id)
                break # Move to the next vector_id once a match is found

    if not ids_to_delete:
        print("No matching vectors found to delete. The index is already clean.")
        return

    print(f"Found {len(ids_to_delete)} vectors to delete.")

    # Delete vectors in batches of 1000 (Pinecone's limit)
    batch_size = 1000
    for i in range(0, len(ids_to_delete), batch_size):
        batch = ids_to_delete[i:i + batch_size]
        try:
            print(f"Deleting batch {i//batch_size + 1}...")
            index.delete(ids=batch)
            print(f"Successfully deleted {len(batch)} vectors.")
        except Exception as e:
            print(f"An error occurred during batch deletion: {e}")

    print("\n--- Deletion Complete ---")
    print(f"Final index stats: {index.describe_index_stats()}")

if __name__ == "__main__":
    main()
