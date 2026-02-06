#!/bin/bash
# 编译LaTeX论文脚本

cd /home/ubuntu/wzz/Cued-Agent/paper

echo "编译论文..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "编译完成！输出文件: main.pdf"
