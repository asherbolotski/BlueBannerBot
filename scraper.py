import time
import os
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth

# --- Configurations for different sites ---
SITE_CONFIGS = {
    'wpilib_docs': {
        'start_url': 'https://docs.wpilib.org/en/stable/docs/software/basic-programming/introduction-to-robot-programming.html', # NEW, simpler start page
        'base_domain': 'https://docs.wpilib.org/en/stable/',
        'output_dir': 'scraped_data_wpilib',
        'content_selector': ('div', {'class_': 'theme-doc-markdown markdown'})
    },
    'javadoc': {
        'start_url': 'https://github.wpilib.org/allwpilib/docs/release/java/edu/wpi/first/wpilibj/package-summary.html',
        'base_domain': 'https://github.wpilib.org/allwpilib/docs/release/java/',
        'output_dir': 'scraped_data_javadoc',
        'content_selector': ('main', {'role': 'main'})
    }
}

# --- CHOOSE YOUR TARGET SITE HERE ---
TARGET_SITE = 'wpilib_docs'
# ------------------------------------

def get_page_source_with_selenium(url):
    """Uses a simple, stable Selenium setup to get the page source."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    # Headless mode is disabled to prevent crashes. A browser window will appear.
    # chrome_options.add_argument("--headless")

    webdriver_service = Service('./chromedriver') 
    driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

    html_source = None
    try:
        print("  -> Getting page in visible browser...")
        driver.get(url)

        # --- Wait for the single most important element: the sidebar with the links ---
        print("  -> Waiting for sidebar navigation...")
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "theme-doc-sidebar-container")))
        # --------------------------------------------------------------------------

        html_source = driver.page_source
        print("  -> Page loaded. Grabbing HTML.")

    except Exception as e:
        print(f"  -> Selenium error: {e}")
    finally:
        driver.quit()

    return html_source

def scrape_page(html_source, content_selector):
    """Parses HTML source to extract text and a BeautifulSoup object."""
    if not html_source:
        return None, None
        
    soup = BeautifulSoup(html_source, 'html.parser')
    tag_name, attrs = content_selector
    content_area = soup.find(tag_name, attrs=attrs)
    if not content_area:
        # Fallback for pages that might not have the main content div
        content_area = soup.find('main') or soup.find('body')

    text = content_area.get_text(separator='\n', strip=True)
    return text, soup

def find_doc_links(soup, current_url, base_domain):
    """Finds all valid, on-domain documentation links on a page."""
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(current_url, href).split('#')[0]
        
        if full_url.startswith(base_domain) and not full_url.endswith(('.zip', '.pdf', '.jar')):
            links.add(full_url)
    return links

def generate_filename(url, base_domain):
    """Creates a safe filename from a URL."""
    filename = url.replace(base_domain, '').replace('/', '_').replace('.html', '') + '.txt'
    return filename

# --- Main script execution ---
if __name__ == "__main__":
    config = SITE_CONFIGS[TARGET_SITE]
    start_url = config['start_url']
    base_domain = config['base_domain']
    output_dir = config['output_dir']
    content_selector = config['content_selector']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    urls_to_visit = deque([start_url])
    visited_urls = set()

    while urls_to_visit:
        current_url = urls_to_visit.popleft()
        if current_url in visited_urls:
            continue

        print(f"Scraping {current_url}...")
        visited_urls.add(current_url)
        
        html = get_page_source_with_selenium(current_url)
        scraped_text, soup = scrape_page(html, content_selector)

        if scraped_text and soup:
            filename = generate_filename(current_url, base_domain)
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(scraped_text)
            print(f"  -> Saved to {file_path}")

            new_links = find_doc_links(soup, current_url, base_domain)
            for link in new_links:
                if link not in visited_urls:
                    urls_to_visit.append(link)
    
    print("\nCrawling complete!")
    print(f"Visited and scraped {len(visited_urls)} pages in '{output_dir}'.")