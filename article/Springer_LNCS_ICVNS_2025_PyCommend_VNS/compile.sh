#!/bin/bash
# Script para compilar pycommend.tex com bibliografia
# Uso: ./compile.sh

export PATH="$PATH:/c/Users/Augusto/AppData/Local/Programs/MiKTeX/miktex/bin/x64"

echo "Compilando pycommend.tex..."
echo "========================================"

echo "[1/4] Primeira compilação com pdflatex..."
pdflatex -interaction=nonstopmode pycommend.tex > /dev/null 2>&1

echo "[2/4] Processando bibliografia com bibtex..."
bibtex pycommend > /dev/null 2>&1

echo "[3/4] Segunda compilação com pdflatex..."
pdflatex -interaction=nonstopmode pycommend.tex > /dev/null 2>&1

echo "[4/4] Terceira compilação com pdflatex..."
pdflatex -interaction=nonstopmode pycommend.tex > /dev/null 2>&1

echo "========================================"
echo "✓ Compilação concluída!"
echo ""
ls -lh pycommend.pdf
echo ""
echo "PDF gerado: pycommend.pdf"
