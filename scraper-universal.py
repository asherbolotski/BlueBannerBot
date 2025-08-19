import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# --- 1. Configuration: Add new websites to this list ---
SITES_TO_SCRAPE = [
    {
    "base_url": "https://docs.limelightvision.io/docs/docs-limelight/getting-started/summary",
    "allowed_domain": "docs.limelightvision.io",
    "output_dir": "limelight_docs_output",
    "content_selector": ("article", {"class": "theme-doc-markdown"}) 
    },
]

REQUEST_DELAY_SECONDS = 1

def scrape_page(url, content_selector):
    """
    Fetches and extracts the main text content from a single page.
    Includes a fallback to scrape the whole body if the primary selector is not found.
    """
    print(f"  - Scraping: {url}")
    try:
        headers = {'User-Agent': 'BlueBannerBot-Scraper/1.0'}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"  - Failed with status code: {response.status_code}")
            return None, None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tag, attrs = content_selector
        main_content = soup.find(tag, attrs=attrs)

        # --- NEW: Fallback Logic ---
        if not main_content:
            print(f"  - WARNING: Main content selector ('{tag}' with attrs {attrs}) not found.")
            print("  - Falling back to scraping the entire <body>.")
            main_content = soup.find('body') # Use the whole body as a fallback

        if main_content:
            # Clean up the content by removing script and style tags
            for element in main_content(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            return main_content.get_text(separator='\n', strip=True), soup
        else:
            # This will only happen if a page has no body tag, which is very rare.
            print("  - ERROR: Could not find any content to scrape.")
            return None, soup 

    except requests.RequestException as e:
        print(f"  - Error during request: {e}")
        return None, None

def crawl_site(config):
    """
    Main crawling logic for a single site configuration.
    """
    base_url = config["base_url"]
    allowed_domain = config["allowed_domain"]
    output_dir = config["output_dir"]
    content_selector = config["content_selector"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    urls_to_visit = {base_url}
    visited_urls = set()

    while urls_to_visit:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        print(f"\nVisiting: {current_url}")
        visited_urls.add(current_url)

        content, soup = scrape_page(current_url, content_selector)

        if soup: 
            if content:
                parsed_url = urlparse(current_url)
                path = parsed_url.path.strip('/')
                filename = path.replace('/', '_').replace('.', '_') + ".txt" if path else "index.txt"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  - Saved content to {filepath}")

            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(current_url, href)
                parsed_absolute_url = urlparse(absolute_url)
                url_without_fragment = parsed_absolute_url._replace(fragment="").geturl()

                if "cdn-cgi/l/email-protection" in url_without_fragment:
                    continue

                if (allowed_domain in url_without_fragment and 
                    url_without_fragment not in visited_urls and
                    url_without_fragment not in urls_to_visit and
                    not any(ext in url_without_fragment for ext in ['.zip', '.pdf', '.png', '.jpg'])):
                    
                    urls_to_visit.add(url_without_fragment)
        
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nFinished crawling {allowed_domain}!")

if __name__ == "__main__":
    for site_config in SITES_TO_SCRAPE:
        print(f"\n{'='*50}\nStarting crawl for: {site_config['allowed_domain']}\n{'='*50}")
        crawl_site(site_config)
