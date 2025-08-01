import requests
from bs4 import BeautifulSoup
import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse

# --- Configuration ---
# The starting point for the crawl on the JAVADOC site
START_URL = "https://github.wpilib.org/allwpilib/docs/release/java/edu/wpi/first/wpilibj/package-summary.html"
# The base domain to stay within, so we don't crawl the entire internet
BASE_DOMAIN = "https://github.wpilib.org/allwpilib/docs/release/java/"
# Directory to store the scraped text files
OUTPUT_DIR = 'scraped_data_javadoc'


def scrape_page(url):
    """Downloads and parses a single page, returning its text and BeautifulSoup object."""
    print(f"Scraping {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # MODIFIED: Javadoc uses a <main> tag for its primary content.
        content_div = soup.find('main', attrs={'role': 'main'})

        if not content_div:
            return None, None

        text = content_div.get_text(separator='\n', strip=True)
        return text, soup
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None, None

def find_doc_links(soup, base_url):
    """Finds all valid, on-domain documentation links on a page."""
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        full_url = full_url.split('#')[0]

        # MODIFIED: Check if the link is within our NEW target domain
        if full_url.startswith(BASE_DOMAIN) and not full_url.endswith(('.zip', '.pdf', '.jar')):
            links.add(full_url)
    return links

def generate_filename(url):
    """Creates a safe filename from a URL."""
    path = urlparse(url).path
    # This logic is complex for Javadocs, let's simplify by replacing slashes
    filename = url.replace(BASE_DOMAIN, '').replace('/', '_') + '.txt'
    return filename

# --- Main script execution ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    urls_to_visit = deque([START_URL])
    visited_urls = set()

    while urls_to_visit:
        current_url = urls_to_visit.popleft()

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        scraped_text, soup = scrape_page(current_url)

        if scraped_text and soup:
            filename = generate_filename(current_url)
            file_path = os.path.join(OUTPUT_DIR, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(scraped_text)
            print(f"  -> Saved to {file_path}")

            new_links = find_doc_links(soup, current_url)
            for link in new_links:
                if link not in visited_urls:
                    urls_to_visit.append(link)
        
        time.sleep(0.1)

    print("\nCrawling complete!")
    print(f"Visited and scraped {len(visited_urls)} pages.")