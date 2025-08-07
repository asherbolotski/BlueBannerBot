import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# --- Configuration ---
# The starting point for the crawl.
BASE_URL = "https://docs.wpilib.org/en/stable/"
# The domain to stay within. Prevents the crawler from leaving the docs site.
ALLOWED_DOMAIN = "docs.wpilib.org"
# Directory to save the scraped text files.
OUTPUT_DIR = "wpilib_docs_output"
# Delay between requests to be respectful to the server.
REQUEST_DELAY_SECONDS = 1

def fetch_page_soup(url):
    """
    Fetches a page and returns the parsed BeautifulSoup object.
    This function makes the single network request for a URL.

    Args:
        url (str): The URL of the page to fetch.

    Returns:
        BeautifulSoup: The soup object if the request is successful, otherwise None.
    """
    print(f"  - Fetching: {url}")
    try:
        # Set a user-agent to identify our bot.
        headers = {'User-Agent': 'FRC-AI-Scraper/1.0'}
        response = requests.get(url, headers=headers, timeout=15)
        
        # Check for a successful request (status code 200)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"  - Failed to fetch page {url} with status code: {response.status_code}")
            return None

    except requests.RequestException as e:
        print(f"  - Error during request for {url}: {e}")
        return None

def main():
    """
    Main function to crawl the website and save the content.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Using a set to avoid visiting the same URL multiple times
    urls_to_visit = {BASE_URL}
    visited_urls = set()

    while urls_to_visit:
        current_url = urls_to_visit.pop()
        
        if current_url in visited_urls:
            continue

        print(f"\nVisiting: {current_url}")
        visited_urls.add(current_url)

        # Fetch the page and get the soup object ONCE.
        soup = fetch_page_soup(current_url)

        # If the fetch was successful (soup is not None)
        if soup:
            # --- 1. Extract and save the text content ---
            # CORRECTED: The main content on WPILib docs is in a div with class="document"
            main_content = soup.find('div', attrs={'class': 'document'})
            
            if main_content:
                print("  - Found main content section. Preparing to save file.")
                # Remove script/style tags for cleaner text
                for element in main_content(["script", "style"]):
                    element.decompose()
                
                content_text = main_content.get_text(separator='\n', strip=True)

                # Create a valid filename from the URL
                parsed_url = urlparse(current_url)
                path = parsed_url.path.strip('/')
                filename = path.replace('/', '_') + ".txt" if path else "index.txt"
                
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content_text)
                print(f"  - Successfully saved content to {filepath}")
            else:
                # Updated diagnostic message for the new selector
                print("  - WARNING: Main content section with class='document' not found. Skipping file save for this URL.")


            # --- 2. Find all new links to visit from the same soup object ---
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Create an absolute URL from a relative one (e.g., "../page.html")
                absolute_url = urljoin(current_url, href)
                
                # Parse the URL to remove fragments (e.g., "#section-name")
                parsed_absolute_url = urlparse(absolute_url)
                url_without_fragment = parsed_absolute_url._replace(fragment="").geturl()

                # Check if the link is within our allowed domain and hasn't been seen
                if (ALLOWED_DOMAIN in url_without_fragment and 
                    url_without_fragment not in visited_urls and
                    url_without_fragment not in urls_to_visit):
                    
                    # Filter out links to files
                    if any(ext in url_without_fragment for ext in ['.zip', '.pdf', '.png', '.jpg']):
                        continue

                    urls_to_visit.add(url_without_fragment)

        # Be a good web citizen!
        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nCrawling finished!")

if __name__ == "__main__":
    main()
