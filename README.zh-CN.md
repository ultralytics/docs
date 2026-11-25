<a href="https://www.ultralytics.com" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics 标志"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 📚 Ultralytics 文档

欢迎来到 Ultralytics 文档，这里是全面了解和使用我们最先进的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工具和模型（包括 [Ultralytics YOLO](https://docs.ultralytics.com/models)）的资源中心。这些文档持续维护，并部署到 [https://docs.ultralytics.com](https://docs.ultralytics.com)，方便随时查阅。

[![Ultralytics Actions](https://github.com/ultralytics/docs/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/format.yml)
[![jsDelivr hits](https://data.jsdelivr.com/v1/package/gh/ultralytics/llm/badge?style=rounded)](https://www.jsdelivr.com/package/gh/ultralytics/llm)

[![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🛠️ Ultralytics 软件包

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

上方徽章跟踪的是本站所文档化的核心 `ultralytics` Python 软件包。若要以开发者模式安装该软件包（可直接修改源代码），请确保系统已安装 [Git](https://git-scm.com/downloads/) 和 [Python](https://www.python.org/downloads/) 3.8 或更高版本，然后按以下步骤操作：

1.  使用 Git 将 `ultralytics` 仓库克隆到本地：

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2.  进入克隆仓库的根目录：

    ```bash
    cd ultralytics
    ```

3.  使用 [pip](https://pip.pypa.io/en/stable/installation/) 以可编辑模式（`-e`）安装软件包及其开发依赖（`[dev]`）：

    ```bash
    pip install -e '.[dev]'
    ```

    该命令安装的 `ultralytics` 软件包会让源代码的改动立即在环境中生效，非常适合开发和贡献代码。

## 🧰 技术栈

- **[Next.js](https://nextjs.org/)** - 渲染已发布文档站点的框架
- **[Ultralytics Chat](https://github.com/ultralytics/llm)** - 实时对话式 AI，开源实现为 [chat.js](https://github.com/ultralytics/llm)
- **[Vercel](https://vercel.com/)** - 托管与部署
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD 自动化

## 🔧 仓库工具

本仓库包含文档源文件以及用于检查已发布站点的维护工作流。在本地运行仓库工具前，请先安装轻量的 Python 依赖：

```bash
uv pip install -r requirements.txt
```

例如，网站链接检查工作流会使用 `utils/check_image_sizes.py` 标记已下载页面中体积过大的图片。

## 🌍 网站检查

[docs.ultralytics.com](https://docs.ultralytics.com) 上的在线文档通过 `.github/workflows/` 中的定时和手动工作流进行维护：

- `links.yml` 下载 `www.ultralytics.com`、`docs.ultralytics.com`、`academy.ultralytics.com` 和 `handbook.ultralytics.com` 的渲染页面，然后检查链接、拼写和大图片。
- `links_local.yml` 检查仓库中 Markdown 和 HTML 的链接，手动运行时可选择进行更广泛的链接扫描。
- `download_websites.yml` 可下载公开网站页面以供检查。

生产环境会将本仓库的内容与其他文档来源合并，使用 `zensical build --strict` 验证完整的文档树，并通过 Ultralytics 的集中发布服务进行渲染。因此本仓库没有独立的站点构建流程；请使用上述工作流检查已发布的页面。

## 💡 贡献

我们非常重视开源社区为改进 Ultralytics 项目所做的贡献。您的参与推动着[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)和[人工智能](https://www.ultralytics.com/glossary/artificial-intelligence-ai)领域的创新！请阅读我们的[贡献指南](https://docs.ultralytics.com/help/contributing)，了解参与方式的详细信息。您也可以通过我们的快速[问卷调查](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享反馈和想法。衷心感谢 🙏 所有贡献者的投入与支持！

[![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

我们期待您的贡献！

## 📜 许可证

Ultralytics 文档提供两种许可选项，以适应不同的使用场景：

- **AGPL-3.0 许可证**：非常适合从事学术研究和开放协作的学生、研究人员和爱好者。完整详情请参阅 [LICENSE](https://github.com/ultralytics/docs/blob/main/LICENSE) 文件。该许可证鼓励将改进回馈社区，营造开放协作的环境。
- **企业许可证**：专为商业应用设计，允许将 Ultralytics 软件和 [AI 模型](https://docs.ultralytics.com/models)无缝集成到商业产品和服务中，无需遵守 AGPL-3.0 的开源要求。如需获取企业许可证，请访问 [Ultralytics 授权许可](https://www.ultralytics.com/license)了解更多信息。

## ✉️ 联系方式

有关文档的错误报告、功能请求和其他问题，请使用 [GitHub Issues](https://github.com/ultralytics/docs/issues)。如需讨论、提问或获取有关 Ultralytics 软件、[Ultralytics Platform](https://docs.ultralytics.com/platform/quickstart) 等方面的社区支持，欢迎加入我们的 [Discord 服务器](https://discord.com/invite/ultralytics)，与同行和 Ultralytics 团队交流！

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
