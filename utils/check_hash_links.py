# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def check_hash_links(download_dir, website):
    """Check for links ending with # in downloaded HTML files."""
    print(f"Scanning {download_dir} for links ending with #...")
    hash_links = {}

    for html_file in Path(download_dir).rglob("*.html"):
        try:
            with open(html_file, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            page_url = (
                f"https://{website}/{html_file.relative_to(download_dir)}".replace("/index.html", "/").removesuffix(".html")
            )

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                # Check if link ends with # but not #anchor (bare # with no anchor name indicates a broken link)
                if href.endswith("#") and href != "#":
                    if href not in hash_links:
                        hash_links[href] = []
                    if page_url not in hash_links[href]:
                        hash_links[href].append(page_url)
        except Exception:
            pass

    if hash_links:
        print(f"\n⚠️ Found {len(hash_links)} links ending with #")
        output = [f"*{len(hash_links)} links ending with #*{' (showing first 10)' if len(hash_links) > 10 else ''}"]
        for link, pages in sorted(hash_links.items())[:10]:
            page_count = f" ({len(pages)} pages)" if len(pages) > 1 else ""
            example_page = pages[0]
            output.append(f"• <{link}|{link}>{page_count} ➜ {example_page}")

        result = "\\n".join(output)
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"HASH_LINKS<<EOF\n{result}\nEOF\n")
        return 1
    else:
        print("✅ No links ending with # found")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_hash_links.py <download_dir> <website>")
        sys.exit(1)

    sys.exit(check_hash_links(sys.argv[1], sys.argv[2]))
