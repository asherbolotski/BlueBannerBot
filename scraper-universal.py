import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# --- 1. Configuration: Add new websites to this list ---
SITES_TO_SCRAPE = [
    {
        # CORRECTED: Updated the content selector for REV Robotics
        "base_url": "https://docs.revrobotics.com/",
        "allowed_domain": "docs.revrobotics.com",
        "output_dir": "rev_docs_output",
        "content_selector": ("div", {"class": "theme-default-content"}) 
    },
    {
        "base_url": "https://api.ctr-electronics.com/phoenix5/api/java/com/ctre/phoenix/package-summary.html",
        "allowed_domain": "api.ctr-electronics.com",
        "output_dir": "ctre_docs_output",
        "content_selector": ("div", {"class": "contentContainer"})
    },
    {
        "base_url": "https://docs.limelightvision.io/docs/v2024/getting-started",
        "allowed_domain": "docs.limelightvision.io",
        "output_dir": "limelight_docs_output",
        "content_selector": ("main", {}) 
    },

]

REQUEST_DELAY_SECONDS = 1

def scrape_page(url, content_selector):
    """
    Fetches and extracts the main text content from a single page.
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

        if main_content:
            for element in main_content(["script", "style"]):
                element.decompose()
            return main_content.get_text(separator='\n', strip=True), soup
        else:
            print(f"  - WARNING: Main content selector ('{tag}' with attrs {attrs}) not found.")
            return None, soup # Return soup anyway to find other links

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

        if soup: # If the page was fetched successfully
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
