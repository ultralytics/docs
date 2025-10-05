# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# URLs to ignore when checking image sizes
URL_IGNORE_LIST = {
    # Add image URLs here that should be excluded from size checks
    # Example: "https://example.com/large-banner.png",
}


def check_image_sizes(download_dir, website, threshold_kb=500, max_workers=32, ignore_gifs=False):
    """Check image sizes in downloaded HTML files and report large images."""
    print(f"Scanning {download_dir} for images...")
    unique_images = defaultdict(set)

    # Scan downloaded HTML files for image URLs
    for html_file in Path(download_dir).rglob("*.html"):
        try:
            with open(html_file, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            page_url = f"https://{website}/{html_file.relative_to(download_dir)}".replace("/index.html", "/")
            for img in soup.find_all("img", src=True):
                img_url = urljoin(f"https://{website}", img["src"])
                if img_url not in URL_IGNORE_LIST:
                    if not ignore_gifs or not img_url.lower().endswith(".gif"):
                        unique_images[img_url].add(page_url)
        except Exception:
            pass

    print(f"Found {len(unique_images)} unique images")

    # Check sizes
    def get_size(url):
        """Get file size and format for a URL."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=10, headers=HEADERS)
            size = int(response.headers.get("content-length", 0))
            if size == 0:
                response = requests.get(url, allow_redirects=True, timeout=10, stream=True, headers=HEADERS)
                size = int(response.headers.get("content-length", 0))
                response.close()

            # Get format from Content-Type header first (more reliable)
            content_type = response.headers.get("content-type", "").lower()
            format_map = {
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/svg+xml": ".svg",
                "image/avif": ".avif",
                "image/bmp": ".bmp",
                "image/tiff": ".tiff",
            }
            fmt = format_map.get(content_type.split(";")[0].strip())

            # Fallback to URL parsing - check original URL first, then redirected URL
            if not fmt:
                fmt = Path(urlparse(url).path).suffix.lower()
                if not fmt:
                    final_url = response.url if response.history else url
                    fmt = Path(urlparse(final_url).path).suffix.lower() or ".unknown"

            return url, size, fmt
        except Exception:
            return url, None, None

    # Collect all image data
    all_images = []
    with requests.Session() as session:
        session.headers.update(HEADERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for url, size, fmt in executor.map(get_size, unique_images.keys()):
                if size:
                    all_images.append((size / 1024, fmt, len(unique_images[url]), url))

    all_images.sort(reverse=True)

    # Print statistics
    if all_images:
        import pandas as pd

        df = pd.DataFrame(all_images, columns=["Size (KB)", "Format", "Pages", "URL"])

        # Format statistics
        format_stats = (
            df.groupby("Format")
            .agg({"URL": "count", "Size (KB)": ["min", "max", "mean", "sum"]})
            .round(1)
            .reset_index()
        )
        format_stats.columns = ["Format", "Count", "Min (KB)", "Max (KB)", "Mean (KB)", "Total (MB)"]
        format_stats["Total (MB)"] = (format_stats["Total (MB)"] / 1024).round(1)
        format_stats = format_stats.sort_values("Total (MB)", ascending=False)

        print("\nImage Format Statistics:")
        print(format_stats.to_string(index=False))
        print(f"\nTotal images processed: {len(all_images)}")

        # Print top 50 largest
        print("\nTop 50 Largest Images:")
        top_50 = df.head(50).copy()
        top_50["Size (KB)"] = top_50["Size (KB)"].round(1)
        top_50["Example Page"] = top_50["URL"].apply(lambda url: list(unique_images[url])[0])
        top_50["URL"] = top_50["URL"].apply(lambda x: x if len(x) <= 120 else x[:60] + "..." + x[-57:])
        print(top_50[["URL", "Pages", "Size (KB)", "Format", "Example Page"]].to_string(index=False))

    # Check for large images above threshold
    large_images = [(size_kb, fmt, pages, url) for size_kb, fmt, pages, url in all_images if size_kb >= threshold_kb]

    if large_images:
        print(f"\n⚠️ Found {len(large_images)} images >= {threshold_kb} KB")
        output = [f"*{len(large_images)} images >= {threshold_kb}KB*{' (showing first 10)' if len(large_images) > 10 else ''}"]
        for size_kb, fmt, pages, url in large_images[:10]:
            # Extract filename from URL for concise display
            filename = Path(urlparse(url).path).name or "image"
            # Append format if filename doesn't have an extension
            if not Path(filename).suffix and fmt:
                filename = f"{filename}{fmt}"
            # Truncate from start if too long, keeping extension visible
            if len(filename) > 40:
                filename = "..." + filename[-37:]
            # Get first page URL for context
            page_url = list(unique_images[url])[0]
            # Format as Slack hyperlink to avoid auto-expansion: <url|text>
            output.append(f"• {size_kb:.0f}KB <{url}|{filename}> ➜ {page_url}")

        result = "\\n".join(output)
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"IMAGE_RESULTS<<EOF\n{result}\nEOF\n")
        return 1
    else:
        print(f"\n✅ No images >= {threshold_kb} KB")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_image_sizes.py <download_dir> <website> [ignore_gifs]")
        sys.exit(1)

    download_dir = sys.argv[1]
    website = sys.argv[2]
    ignore_gifs = sys.argv[3].lower() in ("true", "1", "yes") if len(sys.argv) > 3 else False
    sys.exit(check_image_sizes(download_dir, website, ignore_gifs=ignore_gifs))
