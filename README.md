# Ultralytics Documentation 📘

Welcome to the Ultralytics documentation repository! Our documentation, hosted at [Ultralytics Docs](https://docs.ultralytics.com), provides comprehensive guidelines, tutorials, and API references to ensure a smooth experience with our machine learning solutions.

<!-- Ensure that badges and links remain unchanged -->
[![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment)  [![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)

## Installation Guide 🔧

Install the `ultralytics` package for your ML projects by following these steps, ensuring you have Git and Python 3 installed on your system:

1. **Clone the Repository**: Use Git to download the `ultralytics` repository onto your local machine.
    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2. **Change Directory**: Navigate to the repository's root to access the package content.
    ```bash
    cd ultralytics
    ```

3. **Install in Developer Mode**: Use `pip` to install the package with developer privileges. This allows for real-time updates to the codebase in your environment.
    ```bash
    pip install -e '.[dev]'
    ```
    > 📝 Note: If you have multiple Python versions, you may need to use `pip3`.

## Local Development Workflow 🚀

### Building and Serving Docs Locally

If you're making documentation changes and want to preview them locally, employ the `mkdocs serve` command as follows:
```bash
mkdocs serve
```
- **Build and Preview Locally**: Watch your changes in real-time as you modify your documentation.

To stop the server, simply use the `CTRL+C` interrupt command in your terminal.

### Supporting Multiple Languages

For multi-language documentation, ensure you perform these additional steps:

1. **Commit New Translations**: Add all new translations in Markdown files to the git repository.
    ```bash
    git add docs/**/*.md -f
    ```
2. **Build Translations**: Manually build each language using the respective MkDocs configuration files.
    ```bash
    # Clear the existing /site directory
    rm -rf site

    # Build the primary language
    mkdocs build -f docs/mkdocs.yml

    # Build additional languages
    for file in docs/mkdocs_*.yml; do
      echo "Now building docs with: $file"
      mkdocs build -f "$file"
    done
    ```
3. **Preview the Multi-language Site**: Visually inspect the translations in your browser.
    ```bash
    cd site
    python -m http.server
    # MacOS users can open the site with:
    open http://localhost:8000
    ```

## Deployment 🌐

To publish your MkDocs documentation site, identify a suitable host (e.g., GitHub Pages, GitLab Pages, or Amazon S3), and configure the `mkdocs.yml` with necessary deployment details.

Deploy using the command below (using GitHub Pages as an example):
```bash
mkdocs gh-deploy
```
For GitHub Pages, you can even establish a custom domain in your repository's "Settings."

![GitHub Custom Domain Settings](https://user-images.githubusercontent.com/26833433/210150206-9e86dcd7-10af-43e4-9eb2-9518b3799eac.png)

Refer to [MkDocs Documentation](https://www.mkdocs.org/user-guide/deploying-your-docs/) for additional deployment strategies.

## Contributions & Feedback 🤝

Your contributions make Ultralytics thrive! If you want to contribute or provide feedback, please follow our [Contributing Guide](https://docs.ultralytics.com/help/contributing) and share your insights through our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). We extend our gratitude 🙏 to all our contributors and community members!

![Ultralytics Open-source Contributors](https://github.com/ultralytics/assets/raw/main/im/image-contributors.png)

## Licensing ⚖️

Ultralytics offers two types of licenses:

- **[AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)**: Our open-source license for individuals, researchers, and enthusiasts advocating for widespread knowledge exchange.
- **Enterprise License**: Suited for commercial entities, allowing integration of our software into your products without the obligations of the AGPL-3.0. Interested parties can explore more at [Ultralytics Licensing](https://ultralytics.com/license).

## Community & Support 💬

If you find issues or have feature suggestions, report them at [GitHub Issues](https://github.com/ultralytics/docs/issues). Join our conversations on [Discord](https://ultralytics.com/discord) for vibrant discussions and community support!

<!-- Social Media Links -->
<div align="center">
  <a href="https://github.com/ultralytics">![Ultralytics GitHub](https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png)</a>
  <a href="https://www.linkedin.com/company/ultralytics/">![Ultralytics LinkedIn](https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png)</a>
  <a href="https://twitter.com/ultralytics">![Ultralytics Twitter](https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png)</a>
  <a href="https://youtube.com/ultralytics">![Ultralytics YouTube](https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png)</a>
  <a href="https://www.tiktok.com/@ultralytics">![Ultralytics TikTok](https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png)</a>
  <a href="https://www.instagram.com/ultralytics/">![Ultralytics Instagram](https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png)</a>
  <a href="https://ultralytics.com/discord">![Ultralytics Discord](https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png)</a>
</div>
