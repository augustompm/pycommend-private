#!/bin/bash
# Script de verificação manual do artigo compilado
# Uso: ./verify.sh

echo "======================================"
echo "VERIFICAÇÃO DO ARTIGO PYCOMMEND VNS"
echo "======================================"
echo ""

# Verificar se PDF existe
if [ ! -f "pycommend.pdf" ]; then
    echo "❌ ERRO: pycommend.pdf não encontrado!"
    echo "Execute ./compile.sh primeiro"
    exit 1
fi

# Informações do arquivo
echo "📄 ARQUIVO PDF"
echo "--------------------------------------"
ls -lh pycommend.pdf
echo ""

# Contar páginas (baseado no log)
if [ -f "pycommend.log" ]; then
    pages=$(grep -o '\[.*\]' pycommend.log | tail -1 | grep -o '[0-9]*' | tail -1)
    echo "📖 Páginas: $pages"
fi

# Verificar citações processadas
if [ -f "pycommend.bbl" ]; then
    refs=$(grep -c "\\bibitem" pycommend.bbl)
    echo "📚 Referências na bibliografia: $refs"
fi

# Verificar warnings
if [ -f "pycommend.log" ]; then
    warnings=$(grep -c "Warning" pycommend.log)
    echo "⚠️  Warnings: $warnings (esperado: alguns warnings de formatação)"
fi

echo ""
echo "======================================"
echo "CHECKLIST MANUAL"
echo "======================================"
echo ""
echo "Abra o PDF e verifique:"
echo ""
echo "□ Página 1 - Título"
echo "  ✓ PyCommend VNS: A Multi-Objective Python Library Recommendation Framework"
echo "  ✓ Autores: Augusto M. P. de Mendonça, Filipe P. Sousa, Igor M. Coelho"
echo "  ✓ Afiliações: UFF e UERJ"
echo ""
echo "□ Página 1 - Abstract"
echo "  ✓ ~250 palavras"
echo "  ✓ Menciona 42% tempo dev, \$300B GDP loss"
echo "  ✓ Menciona 641,000+ pacotes PyPI"
echo "  ✓ Menciona MOVNS com 3 objetivos (LU, SS, RSS)"
echo "  ✓ Menciona resultados: 5.1% melhor HV, 15.2% convergência"
echo ""
echo "□ Página 1 - Keywords"
echo "  ✓ Multi-Objective Optimization"
echo "  ✓ Variable Neighborhood Search"
echo "  ✓ Package Recommendation"
echo "  ✓ Software Reuse, Library Discovery, Python Ecosystem"
echo "  ✓ MOVNS, Pareto Optimization"
echo ""
echo "□ Páginas 2-3 - Introduction"
echo "  ✓ Sem subsections (LNCS compliant)"
echo "  ✓ ~800 palavras"
echo "  ✓ Citações: auch2024, xu2020, ouni2017"
echo "  ✓ Contexto do problema"
echo "  ✓ Contribuições do trabalho"
echo "  ✓ Roadmap das seções"
echo ""
echo "□ Páginas 3-4 - Seções 2-7"
echo "  ✓ TODO markers presentes"
echo "  ✓ Estrutura completa (7 seções, 24 subseções)"
echo ""
echo "□ Página 5 - Bibliografia"
echo "  ✓ 18 referências listadas"
echo "  ✓ Formato Springer LNCS (splncs04)"
echo "  ✓ Primeira referência: Auch et al. 2024"
echo ""
echo "□ Formatação Geral"
echo "  ✓ Fonte: Computer Modern (LaTeX padrão)"
echo "  ✓ Margens: LNCS padrão"
echo "  ✓ Running heads: título abreviado"
echo "  ✓ Numeração de páginas"
echo ""
echo "======================================"
echo "ABRIR PDF"
echo "======================================"
echo ""

# Detectar sistema operacional e abrir PDF
if command -v explorer.exe &> /dev/null; then
    echo "Abrindo PDF no Windows Explorer..."
    explorer.exe pycommend.pdf
elif command -v open &> /dev/null; then
    echo "Abrindo PDF no macOS..."
    open pycommend.pdf
elif command -v xdg-open &> /dev/null; then
    echo "Abrindo PDF no Linux..."
    xdg-open pycommend.pdf
else
    echo "⚠️  Não foi possível abrir automaticamente."
    echo "Abra manualmente: pycommend.pdf"
fi

echo ""
echo "======================================"
echo "ARQUIVOS DE VERIFICAÇÃO"
echo "======================================"
echo ""
echo "Logs disponíveis para análise:"
echo "  - pycommend.log (compilação completa)"
echo "  - pycommend.blg (processamento bibliografia)"
echo "  - pycommend.aux (referências cruzadas)"
echo ""

# Mostrar últimas linhas do log de compilação
echo "Últimas linhas do log de compilação:"
echo "--------------------------------------"
tail -5 pycommend.log
echo ""

echo "✓ Verificação concluída!"
echo ""
echo "Para recompilar: ./compile.sh"
