# Essay (LaTeX / Overleaf)

Implementation Plan, Task List and Thought in Chinese：本目录提供一个满足 `essay/requirements.md` 格式约束的 LaTeX 骨架，可直接上传到 Overleaf 渲染。

## 使用方式（Overleaf）
1. 将 `essay/` 目录中的文件打包上传到 Overleaf（或只上传 `main.tex`、`sections/`、`references.bib`）。
2. 在 Overleaf 中把 `main.tex` 设为主文件（Main document）。
3. 默认用 **pdfLaTeX** 编译即可。

## 字体说明（Times New Roman）
- 目前模板使用 `newtxtext/newtxmath`，在 Overleaf 上稳定、外观接近 Times New Roman。
- 如果课程/格式要求必须是“系统字体的 Times New Roman（严格一致）”，通常需要 **XeLaTeX + fontspec**，并确保编译环境里有 Times New Roman 字体（Overleaf 默认不一定提供）。建议做法是：
  - 本地编译（系统已装 Times New Roman），或
  - 上传可替代的开源 Times 风格字体（例如 TeX Gyre Termes）并在 `fontspec` 中显式指定。

## 参考文献
- 当前使用 `natbib` + BibTeX：编辑 `references.bib`，在正文中用 `\\citet{}` / `\\citep{}`。
- 期刊/会议有特定样式时，再替换 `\\bibliographystyle{...}` 即可。

## 提交为 .docx（课程要求）
要求文件名形如 `studentID_name_paper.docx`。LaTeX 最终转 docx 常见方案：
- 先在 Overleaf 下载 PDF，再用 Word 打开 PDF 另存为 docx（排版可能需要手工微调）。
- 或用 `pandoc` 将 LaTeX 转 docx（对公式/引用可能不完美，仍需人工检查）。

## 你需要填写的内容
- `main.tex`：标题、作者信息、Abstract（150–250 words）。
- `sections/06_ai_declaration.tex`：AI 写作声明（每人独立提交时也要保留）。
- `sections/07_contributions.tex`：个人贡献声明（每人独立提交版本要写清楚）。

