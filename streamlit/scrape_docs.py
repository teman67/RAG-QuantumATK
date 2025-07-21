# scrape_docs.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

BASE_URL = "https://docs.quantumatk.com/manual/Introduction.html"

def get_all_links(base_url):
    visited = set()
    to_visit = [base_url]
    all_pages = []

    while to_visit:
        url = to_visit.pop()
        if url in visited or not url.startswith("https://docs.quantumatk.com/manual/Introduction.html"):
            continue
        visited.add(url)
        try:
            res = requests.get(url)
            soup = BeautifulSoup(res.content, "html.parser")
            all_pages.append((url, soup.get_text()))

            for a_tag in soup.find_all("a", href=True):
                new_url = urljoin(url, a_tag["href"])
                if new_url not in visited:
                    to_visit.append(new_url)
        except Exception as e:
            print(f"Error accessing {url}: {e}")
    return all_pages

def save_docs(pages, output_dir="quantumatk_docs"):
    os.makedirs(output_dir, exist_ok=True)
    for i, (url, text) in enumerate(pages):
        with open(f"{output_dir}/doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n{text}")

if __name__ == "__main__":
    pages = get_all_links(BASE_URL)
    save_docs(pages)
