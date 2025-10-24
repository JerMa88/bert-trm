#!/usr/bin/env python3
"""
Download a sample of Common Crawl data for BERT training.
This downloads a small WARC file from Common Crawl and extracts text.
"""

import os
import gzip
import requests
from pathlib import Path
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
import re


def clean_text(text):
    """Clean extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove very short lines
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20]
    return '\n'.join(lines)


def download_and_extract_warc(warc_url, output_file, max_documents=10000):
    """
    Download a WARC file from Common Crawl and extract text

    Args:
        warc_url: URL to the WARC file
        output_file: Path to save extracted text
        max_documents: Maximum number of documents to extract
    """
    print(f"Downloading Common Crawl data from: {warc_url}")

    # Download WARC file
    response = requests.get(warc_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    documents_extracted = 0

    with open(output_file, 'w', encoding='utf-8') as outf:
        # Stream the WARC file and extract text
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            # Common Crawl WARCs are gzipped
            with gzip.GzipFile(fileobj=response.raw) as gz:
                for record in ArchiveIterator(gz):
                    if record.rec_type == 'response':
                        # Get the content
                        content = record.content_stream().read()

                        try:
                            # Try to decode as text
                            text = content.decode('utf-8', errors='ignore')

                            # Simple HTML stripping (basic)
                            text = re.sub(r'<[^>]+>', ' ', text)
                            text = re.sub(r'&[a-z]+;', ' ', text)

                            # Clean text
                            text = clean_text(text)

                            if len(text) > 100:  # Only keep substantial text
                                outf.write(text + '\n\n')
                                documents_extracted += 1

                                if documents_extracted >= max_documents:
                                    print(f"\nExtracted {documents_extracted} documents")
                                    return

                                if documents_extracted % 100 == 0:
                                    print(f"\rExtracted {documents_extracted} documents...", end='')

                        except Exception as e:
                            # Skip documents that can't be decoded
                            pass

                    pbar.update(len(record.raw_stream.read()))

    print(f"\nTotal documents extracted: {documents_extracted}")


def main():
    # Use a small WARC file from Common Crawl
    # This is from CC-MAIN-2024-10 crawl (March 2024)
    # We'll use just one segment for a manageable download (~1GB compressed)
    warc_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947473735.7/warc/CC-MAIN-20240215104848-20240215134848-00000.warc.gz"

    output_dir = Path("data/common_crawl")
    output_file = output_dir / "crawl_text.txt"

    print("="*70)
    print("Common Crawl Data Download")
    print("="*70)
    print(f"Output: {output_file}")
    print(f"Max documents: 10000")
    print()

    download_and_extract_warc(warc_url, output_file, max_documents=10000)

    print(f"\n✓ Text extracted to: {output_file}")

    # Print file size
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")

    # Count lines
    with open(output_file, 'r') as f:
        lines = sum(1 for _ in f)
    print(f"  Total lines: {lines:,}")


if __name__ == "__main__":
    main()
