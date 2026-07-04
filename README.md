<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 📚 Ultralytics Docs

Welcome to Ultralytics Docs, your comprehensive resource for understanding and utilizing our state-of-the-art [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tools and models, including [Ultralytics YOLO](https://docs.ultralytics.com/models/). These documents are actively maintained and deployed to [https://docs.ultralytics.com](https://docs.ultralytics.com/) for easy access.

[![Ultralytics Actions](https://github.com/ultralytics/docs/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/format.yml)
[![jsDelivr hits](https://data.jsdelivr.com/v1/package/gh/ultralytics/llm/badge?style=rounded)](https://www.jsdelivr.com/package/gh/ultralytics/llm)

[![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment)
[![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)
[![Check Domains](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🛠️ Ultralytics Package

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

The badges above track the core `ultralytics` Python package documented by this site. To install that package in developer mode, which allows you to modify the source code directly, ensure you have [Git](https://git-scm.com/downloads/) and [Python](https://www.python.org/downloads/) 3.8 or later installed on your system. Then, follow these steps:

1.  Clone the `ultralytics` repository to your local machine using Git:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2.  Navigate to the cloned repository's root directory:

    ```bash
    cd ultralytics
    ```

3.  Install the package in editable mode (`-e`) along with its development dependencies (`[dev]`) using [pip](https://pip.pypa.io/en/stable/installation/):

    ```bash
    pip install -e '.[dev]'
    ```

    This command installs the `ultralytics` package such that changes to the source code are immediately reflected in your environment, ideal for development and contributing.

## 🧰 Technology Stack

- **[MkDocs](https://www.mkdocs.org/)** - Static site generator for project documentation
- **[Ultralytics Chat](https://github.com/ultralytics/llm)** - Realtime conversational AI with open-source [chat.js](https://github.com/ultralytics/llm) implementation
- **[GitHub Pages](https://pages.github.com/)** - Hosting and deployment
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automation

## 🔧 Repository Tooling

This repository contains selected documentation Markdown plus maintenance workflows for downloaded website checks, domain checks, and sitemap submissions. Install the lightweight Python dependencies before running repository utilities locally:

```bash
pip install -r requirements.txt
```

For example, `utils/check_image_sizes.py` is used by the website link-check workflow to flag oversized images in downloaded pages.

## 🌍 Website Checks

The live documentation at [docs.ultralytics.com](https://docs.ultralytics.com/) is maintained with scheduled and manual workflows in `.github/workflows/`:

- `links.yml` downloads rendered pages for `www.ultralytics.com`, `docs.ultralytics.com`, and `handbook.ultralytics.com`, then checks links, spelling, and large images.
- `links_local.yml` checks repository Markdown and HTML links, with an optional broader link scan on manual runs.
- `check_domains.yml` verifies Ultralytics domain redirects.
- `download_websites.yml` can download public website pages for inspection.
- `sitemaps.yml` submits sitemaps and changed URLs after successful Pages deployments.

This repository does not currently include local `mkdocs*.yml` configuration files, so README instructions should use the workflows above rather than local MkDocs build commands.

## 💡 Contribute

We deeply value contributions from the open-source community to enhance Ultralytics projects. Your input helps drive innovation in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) and [AI](https://www.ultralytics.com/glossary/artificial-intelligence-ai)! Please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for detailed information on how to get involved. You can also share your feedback and ideas through our quick [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 to all our contributors for their dedication and support!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

We look forward to your contributions!

## 📜 License

Ultralytics Docs are available under two licensing options to accommodate different usage scenarios:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts involved in academic pursuits and open collaboration. See the [LICENSE](https://github.com/ultralytics/docs/blob/main/LICENSE) file for full details. This license promotes sharing improvements back with the community, fostering an open and collaborative environment.
- **Enterprise License**: Designed for commercial applications, this license allows seamless integration of Ultralytics software and [AI models](https://docs.ultralytics.com/models) into commercial products and services without the open-source requirements of AGPL-3.0. Visit [Ultralytics Licensing](https://www.ultralytics.com/license) for more information on obtaining an Enterprise License.

## ✉️ Contact

For bug reports, feature requests, and other issues related to the documentation, please use [GitHub Issues](https://github.com/ultralytics/docs/issues). For discussions, questions, and community support regarding Ultralytics software, the [Ultralytics Platform](https://docs.ultralytics.com/platform/quickstart), and more, join the conversation with peers and the Ultralytics team on our [Discord server](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
