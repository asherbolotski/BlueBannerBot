# First, ensure you have the library installed:
# pip install langchain

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits a given text into smaller chunks using a recursive character splitter.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks
                             to maintain context.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        print("Warning: Input text is empty. Returning an empty list.")
        return []

    # Initialize the text splitter.
    # The default separators ["\n\n", "\n", " ", ""] are excellent for
    # general text and technical documentation. It tries to split by
    # paragraphs first, then lines, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # This is the default, but it's good to be explicit.
    )

    # The create_documents method wraps the text chunks in a Document object,
    # which can be useful for downstream tasks. We will extract the text content.
    # We pass a list containing our single text block.
    documents = text_splitter.create_documents([text])
    
    # Extract the actual text content from the Document objects
    chunks = [doc.page_content for doc in documents]
    
    return chunks

def process_scraped_files():
    """
    This function reads all .txt files from the scraper's output directory,
    chunks them, and prints the results.
    """
    # This should be the same directory your scraper saved files into.
    input_directory = "wpilib_docs_output"

    if not os.path.exists(input_directory):
        print(f"Error: Directory '{input_directory}' not found.")
        print("Please run the web scraper script first to generate the text files.")
        return

    # Loop through all files in the specified directory
    for filename in os.listdir(input_directory):
        # Process only .txt files
        if filename.endswith(".txt"):
            filepath = os.path.join(input_directory, filename)
            print(f"\n--- Processing file: {filename} ---")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    scraped_text = f.read()
                
                if not scraped_text.strip():
                    print("  - File is empty, skipping.")
                    continue

                # Chunk the text from the file
                # You can tune chunk_size and chunk_overlap here.
                text_chunks = chunk_text(scraped_text, chunk_size=1000, chunk_overlap=200)

                print(f"  - Original text length: {len(scraped_text)} characters")
                print(f"  - Total chunks created: {len(text_chunks)}\n")

                # You can uncomment the following lines to see the actual chunks
                # for i, chunk in enumerate(text_chunks):
                #     print(f"--- CHUNK {i+1} (Length: {len(chunk)}) ---")
                #     print(chunk)
                #     print("\n" + "="*40 + "\n")

            except Exception as e:
                print(f"  - Error processing file {filename}: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    process_scraped_files()

