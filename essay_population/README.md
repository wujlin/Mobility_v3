# ICML 2026 Paper (LaTeX)

本目录用于撰写投稿 ICML 2026 的论文（route generation 主线）。

## 快速开始

- 默认无需 ICML 模板也能编译（单栏草稿模式）：`main.tex`
- 当你把官方模板文件放入本目录后，会自动切换到 ICML 模式：
  - `icml2026.sty`（必需）
  - `icml2026.bst`（可选；否则回退到 `plainnat`）

## 本地编译（草稿模式）

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 目录结构

```text
essay_population/
  main.tex
  references.bib
  sections/
    01_introduction.tex
    02_related_work.tex
    03_problem_setup.tex
    04_method.tex
    05_experiments.tex
    06_results.tex
    07_discussion.tex
    08_conclusion.tex
    A_appendix.tex
```

