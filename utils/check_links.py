# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup


def check_links(download_dir, website):
    """Check for circular links and href='#' in downloaded HTML files."""
    print(f"Scanning {download_dir} for problematic links...")
    circular_links = defaultdict(set)
    empty_hrefs = defaultdict(set)

    # Scan downloaded HTML files
    for html_file in Path(download_dir).rglob("*.html"):
        try:
            with open(html_file, encoding="utf-8") as f:
                content = f.read()
                soup = BeautifulSoup(content, "html.parser")

            # Construct page URL
            page_url = f"https://{website}/{html_file.relative_to(download_dir)}".replace(
                "/index.html", "/"
            ).removesuffix(".html")
            page_url_normalized = page_url.rstrip("/")

            # Check for href="#"
            if 'href="#"' in content:
                empty_hrefs[page_url].add('href="#"')

            # Check for circular links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Resolve relative URLs
                absolute_url = urljoin(page_url, href)
                # Normalize by removing trailing slash and hash
                link_url_normalized = absolute_url.split("#")[0].rstrip("/")

                # Check if link points to same page
                if link_url_normalized == page_url_normalized and href not in ("#", ""):
                    circular_links[page_url].add(href)

        except Exception:
            pass

    # Report results
    issues_found = False
    output = []

    if empty_hrefs:
        issues_found = True
        count = sum(len(v) for v in empty_hrefs.values())
        print(f"\n⚠️ Found {count} href='#' in {len(empty_hrefs)} pages")
        output.append(f"*{count} href='#' found on {len(empty_hrefs)} pages*")
        for page_url in sorted(empty_hrefs.keys())[:10]:
            output.append(f"• {page_url}")
        if len(empty_hrefs) > 10:
            output[-1] = output[-1].replace("*", "* (showing first 10)", 1)

    if circular_links:
        issues_found = True
        count = sum(len(v) for v in circular_links.values())
        print(f"\n⚠️ Found {count} circular links on {len(circular_links)} pages")
        if output:
            output.append("")
        output.append(f"*{count} circular links on {len(circular_links)} pages*")
        for page_url in sorted(circular_links.keys())[:10]:
            examples = ", ".join(f"`{h}`" for h in sorted(circular_links[page_url])[:3])
            output.append(f"• {page_url} → {examples}")
        if len(circular_links) > 10:
            output[-1] = output[-1].replace("*", "* (showing first 10)", 1)

    if issues_found:
        result = "\\n".join(output)
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"LINK_ISSUES<<EOF\n{result}\nEOF\n")
        return 1
    else:
        print("\n✅ No problematic links found")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_links.py <download_dir> <website>")
        sys.exit(1)

    download_dir = sys.argv[1]
    website = sys.argv[2]
    sys.exit(check_links(download_dir, website))
