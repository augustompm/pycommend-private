# PyCommend VNS - Instruções de Compilação LaTeX

## Compilador Instalado

**MiKTeX 24.1** instalado via winget em:
```
C:\Users\Augusto\AppData\Local\Programs\MiKTeX\miktex\bin\x64\
```

## Método 1: Script Automático (Recomendado)

```bash
cd /e/pycommend/article/Springer_LNCS_ICVNS_2025_PyCommend_VNS
./compile.sh
```

O script executa automaticamente:
1. pdflatex pycommend.tex
2. bibtex pycommend
3. pdflatex pycommend.tex (2x)

## Método 2: Manual

```bash
cd /e/pycommend/article/Springer_LNCS_ICVNS_2025_PyCommend_VNS

# Adicionar MiKTeX ao PATH
export PATH="$PATH:/c/Users/Augusto/AppData/Local/Programs/MiKTeX/miktex/bin/x64"

# Compilar
pdflatex pycommend.tex
bibtex pycommend
pdflatex pycommend.tex
pdflatex pycommend.tex
```

## Arquivos Gerados

Após compilação bem-sucedida:
- **pycommend.pdf** (154KB, 5 páginas) - Artigo final
- pycommend.aux - Arquivo auxiliar
- pycommend.bbl - Bibliografia processada
- pycommend.blg - Log do BibTeX
- pycommend.log - Log completo da compilação

## Compatibilidade com Overleaf

O template Springer LNCS e arquivo .bib são **100% compatíveis com Overleaf**:
- Template: `llncs.cls` (Springer LNCS v2.24)
- Estilo bib: `splncs04.bst` (Springer)
- Compilador: pdfLaTeX (mesmo do Overleaf)

## Estrutura do Artigo

```latex
\documentclass[runningheads]{llncs}

% Pacotes
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}

% Conteúdo
\begin{document}
  \title{...}
  \author{...}
  \institute{...}
  \maketitle

  \begin{abstract}...\end{abstract}
  \keywords{...}

  \section{Introduction}
  ...

  \bibliographystyle{splncs04}
  \bibliography{pycommend}
\end{document}
```

## Bibliografia

18 referências em `pycommend.bib`:
- 6 Software Library (auch2024, xu2020, ouni2017, thung2013, xie2006, harman2001)
- 5 Multi-objective EA (deb2002, zhang2007, zitzler2003, coello2007, miettinen1999)
- 3 VNS (dahite2022, arroyo2011, hansen2010)
- 2 Embeddings (reimers2019, devlin2019)
- 1 Recent MOEA (liu2024)
- 2 Metrics (while2006, zitzler2007)

## Warnings Conhecidos (Não-críticos)

Durante compilação aparecem alguns warnings esperados:
- **Overfull \hbox**: Linhas ligeiramente longas (ajuste automático do LaTeX)
- **Underfull \vbox**: Espaçamento vertical (normal em LNCS)
- **MiKTeX updates**: Aviso sobre atualizações (pode ignorar)

Nenhum desses warnings impede a geração correta do PDF.

## Visualização do PDF

Abrir arquivo gerado:
```bash
# Windows Explorer
explorer pycommend.pdf

# Ou copiar para Desktop
cp pycommend.pdf ~/Desktop/
```

## Troubleshooting

### Erro: "pdflatex command not found"
```bash
# Adicionar PATH temporariamente
export PATH="$PATH:/c/Users/Augusto/AppData/Local/Programs/MiKTeX/miktex/bin/x64"

# Ou adicionar permanentemente ao ~/.bashrc
echo 'export PATH="$PATH:/c/Users/Augusto/AppData/Local/Programs/MiKTeX/miktex/bin/x64"' >> ~/.bashrc
source ~/.bashrc
```

### Citações aparecem como "?"
Executar o ciclo completo de compilação (pdflatex → bibtex → pdflatex → pdflatex)

### Fontes faltando
MiKTeX baixa fontes automaticamente na primeira compilação (pode demorar alguns minutos)

---

**MiKTeX instalado em**: 2025-09-30
**Versão**: 24.1
**Template**: Springer LNCS v2.24
**Compatível com**: Overleaf ✓
