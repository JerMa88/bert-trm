#!/usr/bin/env python3
"""
Download Wikipedia text data for BERT training.
Uses the Wikipedia API to fetch random articles.
"""

import requests
import time
import re
from pathlib import Path
from tqdm import tqdm


def clean_wiki_text(text):
    """Clean Wikipedia text"""
    # Remove references like [1], [2]
    text = re.sub(r'\[\d+\]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)

    return text.strip()


def fetch_random_article():
    """Fetch a random Wikipedia article"""
    url = "https://en.wikipedia.org/w/api.php"

    # Get random page
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,  # Main namespace only
        "rnlimit": 1
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    if 'query' not in data or 'random' not in data['query']:
        return None

    page_id = data['query']['random'][0]['id']

    # Get page content
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "pageids": page_id,
        "explaintext": True,
        "exsectionformat": "plain"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    if 'query' not in data or 'pages' not in data['query']:
        return None

    page = data['query']['pages'][str(page_id)]
    title = page.get('title', '')
    text = page.get('extract', '')

    if not text or len(text) < 100:
        return None

    return title, clean_wiki_text(text)


def download_wiki_articles(output_file, num_articles=1000):
    """
    Download Wikipedia articles

    Args:
        output_file: Path to save articles
        num_articles: Number of articles to download
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    articles_downloaded = 0
    total_chars = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        with tqdm(total=num_articles, desc="Downloading articles") as pbar:
            while articles_downloaded < num_articles:
                try:
                    result = fetch_random_article()

                    if result:
                        title, text = result

                        # Write article
                        f.write(f"=== {title} ===\n")
                        f.write(text + '\n\n')
                        f.flush()

                        articles_downloaded += 1
                        total_chars += len(text)
                        pbar.update(1)

                        # Be nice to Wikipedia servers
                        time.sleep(0.1)

                    else:
                        # Failed to fetch, wait a bit longer
                        time.sleep(1)

                except Exception as e:
                    print(f"\nError fetching article: {e}")
                    time.sleep(1)

    return articles_downloaded, total_chars


def main():
    output_dir = Path("data/wikipedia")
    output_file = output_dir / "wiki_text.txt"

    print("="*70)
    print("Wikipedia Text Download for BERT Training")
    print("="*70)
    print(f"Output: {output_file}")
    print(f"Target: 1000 articles")
    print()

    articles, chars = download_wiki_articles(output_file, num_articles=1000)

    print(f"\n✓ Downloaded {articles} articles")
    print(f"  Total characters: {chars:,}")

    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")

    # Count lines
    with open(output_file, 'r') as f:
        lines = sum(1 for line in f if line.strip())
    print(f"  Total lines: {lines:,}")


if __name__ == "__main__":
    main()
