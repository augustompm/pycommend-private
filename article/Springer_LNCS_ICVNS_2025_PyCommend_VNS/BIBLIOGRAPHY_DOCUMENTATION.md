# PyCommend VNS - Complete Bibliography Documentation

**Total References**: 18
**Last Updated**: 2025-01-XX
**Article**: PyCommend VNS: A Multi-Objective Python Library Recommendation Framework

---

## 1. SOFTWARE LIBRARY CLASSIFICATION & RECOMMENDATION (6 references)

### [1] auch2024 - Towards an Automated Classification of Software Libraries

**Citation**:
```bibtex
@article{auch2024,
  author = {Auch, Maximilian and Balluf, Maximilian and Mandl, Peter and Wolf, Christian},
  title = {Towards an Automated Classification of Software Libraries},
  journal = {SN Computer Science},
  year = {2024},
  volume = {5},
  number = {339},
  doi = {10.1007/s42979-024-02654-2}
}
```

**Abstract**:
Develops automated classification approach for software libraries using hybrid method combining id-based and tag-based features. Achieves 94.12% classification accuracy on Maven ecosystem. Demonstrates that even in organized ecosystems like Maven, only 75% of libraries have tags and quality varies significantly.

**Relevance to PyCommend**:
- Motivates need for automated library discovery systems
- Shows limitations of existing categorization approaches
- Validates that PyPI (less organized than Maven) has even greater discovery challenges
- Supports problem statement in Section 1

---

### [2] xu2020 - Why Reinventing the Wheels? An Empirical Study on Library Reuse and Re-implementation

**Citation**:
```bibtex
@article{xu2020,
  author = {Xu, Bowen and Ye, Deheng and Xing, Zhenchang and Xia, Xin and Chen, Guibin and Li, Shanping},
  title = {Why reinventing the wheels? An empirical study on library reuse and re-implementation},
  journal = {Empirical Software Engineering},
  year = {2020},
  volume = {25},
  pages = {755--789},
  doi = {10.1007/s10664-019-09771-0}
}
```

**Abstract**:
Identifies three primary barriers to effective library reuse: knowledge barriers (limited awareness of existing solutions), psychological barriers (preference for building from scratch), and technical barriers (difficulty evaluating quality and managing dependencies). Empirically demonstrates the phenomenon of "reinventing the wheel" where developers implement functionality that already exists in established libraries.

**Relevance to PyCommend**:
- Establishes fundamental problem that PyCommend addresses
- Validates economic impact of poor library discovery ($300B GDP loss)
- Supports introduction motivation
- Cited in Section 1 for problem contextualization

---

### [3] ouni2017 - Search-based Software Library Recommendation Using Multi-objective Optimization

**Citation**:
```bibtex
@article{ouni2017,
  author = {Ouni, Ali and Kessentini, Marouane and Inoue, Katsuro and Cinneide, Mel O},
  title = {Search-based software library recommendation using multi-objective optimization},
  journal = {Information and Software Technology},
  year = {2017},
  volume = {83},
  pages = {55--75},
  doi = {10.1016/j.infsof.2016.11.007}
}
```

**Abstract**:
Demonstrates that multi-objective optimization can balance multiple criteria in library selection including functionality matching, license compatibility, and API stability using NSGA-II. However, does not incorporate real-world co-occurrence patterns from dependency files or semantic similarity between packages.

**Relevance to PyCommend**:
- Establishes multi-objective optimization as valid approach for library recommendation
- Identifies gap: lack of co-occurrence patterns and semantic similarity
- PyCommend addresses these limitations by combining usage patterns from GitHub and SBERT embeddings
- Cited in Section 1 and Related Work

---

### [4] thung2013 - Automated Library Recommendation

**Citation**:
```bibtex
@inproceedings{thung2013,
  author = {Thung, Ferdian and Lo, David and Lawall, Julia},
  title = {Automated library recommendation},
  booktitle = {2013 20th Working Conference on Reverse Engineering (WCRE)},
  year = {2013},
  pages = {182--191},
  doi = {10.1109/WCRE.2013.6671293},
  publisher = {IEEE}
}
```

**Abstract**:
Proposes LibRec technique that recommends libraries based on current application dependencies using hybrid approach combining association rule mining and collaborative filtering. Association rules extract library usage patterns, while collaborative filtering recommends libraries used by similar projects. Addresses the challenge that developers are often unaware of suitable libraries for their projects.

**Relevance to PyCommend**:
- Validates hybrid approach combining multiple data sources
- Association rule mining analogous to PyCommend's co-occurrence (Linked Usage objective)
- Collaborative filtering analogous to semantic similarity patterns
- Cited in Related Work for library recommendation systems

---

### [5] xie2006 - MAPO: Mining API Usages from Open Source Repositories

**Citation**:
```bibtex
@inproceedings{xie2006,
  author = {Xie, Tao and Pei, Jian},
  title = {{MAPO}: Mining {API} usages from open source repositories},
  booktitle = {2006 International Workshop on Mining Software Repositories (MSR '06)},
  year = {2006},
  pages = {54--57},
  doi = {10.1145/1137983.1137997},
  publisher = {ACM}
}
```

**Abstract**:
Develops MAPO framework to mine frequent API usage patterns from open source repositories. Given a query describing a method, class, or package, MAPO gathers relevant source files and conducts data mining to produce frequent API usage patterns. Addresses the challenge that complex APIs are often poorly documented, creating barriers for developers using them in new code.

**Relevance to PyCommend**:
- Establishes mining source repositories as valid methodology
- Analogous to PyCommend mining 24,000 GitHub projects for dependency patterns
- Validates extraction of co-occurrence from real-world projects
- Cited in Related Work and Problem Formulation (Section 3)

---

### [6] harman2001 - Search-based Software Engineering

**Citation**:
```bibtex
@article{harman2001,
  author = {Harman, Mark and Jones, Bryan F.},
  title = {Search-based software engineering},
  journal = {Information and Software Technology},
  year = {2001},
  volume = {43},
  number = {14},
  pages = {833--839},
  doi = {10.1016/S0950-5849(01)00189-6}
}
```

**Abstract**:
Foundational paper introducing Search-Based Software Engineering (SBSE) as new field. Argues that software engineering is ideal for metaheuristic search techniques like genetic algorithms, simulated annealing, and tabu search. These techniques provide solutions to difficult problems of balancing competing and sometimes inconsistent constraints, finding acceptable solutions where perfect solutions are theoretically impossible or practically infeasible.

**Relevance to PyCommend**:
- Establishes theoretical foundation for applying metaheuristics to software engineering problems
- Validates multi-objective optimization for library recommendation
- VNS and MOEA/D are metaheuristics addressing competing constraints (LU vs SS vs RSS)
- Cited in Related Work for foundational SBSE concepts

---

## 2. MULTI-OBJECTIVE EVOLUTIONARY ALGORITHMS (5 references)

### [7] deb2002 - A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II

**Citation**:
```bibtex
@article{deb2002,
  author = {Deb, Kalyanmoy and Pratap, Aravind and Agarwal, Sameer and Meyarivan, T.},
  title = {A fast and elitist multiobjective genetic algorithm: {NSGA-II}},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = {2002},
  volume = {6},
  number = {2},
  pages = {182--197},
  doi = {10.1109/4235.996017}
}
```

**Abstract**:
Proposes NSGA-II with fast non-dominated sorting, crowding distance for diversity preservation, and elitism. Achieves better spread of solutions and convergence near true Pareto-optimal front compared to PAES and SPEA. Introduces mechanisms that became standard in evolutionary multi-objective optimization.

**Relevance to PyCommend**:
- Baseline algorithm for experimental comparison
- PyCommend experimental results show MOVNS achieves 5.1% higher HV than NSGA-II
- Non-dominated sorting used internally in MOVNS archive management
- Cited in Related Work and Experimental Setup (Sections 2 and 5)

---

### [8] zhang2007 - MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition

**Citation**:
```bibtex
@article{zhang2007,
  author = {Zhang, Qingfu and Li, Hui},
  title = {{MOEA/D}: A Multiobjective Evolutionary Algorithm Based on Decomposition},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = {2007},
  volume = {11},
  number = {6},
  pages = {712--731},
  doi = {10.1109/TEVC.2007.892759}
}
```

**Abstract**:
Decomposes multi-objective optimization problem into scalar optimization subproblems using aggregation functions (weighted sum, Tchebycheff, or boundary intersection). Each subproblem optimized simultaneously by evolving population of solutions. Neighborhood relations between subproblems exploit information from adjacent weight vectors. Demonstrates superior performance on continuous problems.

**Relevance to PyCommend**:
- Primary baseline for comparison against MOVNS
- PyCommend results show MOVNS achieves 77.6% of MOEA/D performance (competitive)
- Decomposition approach contrasted with VNS neighborhood exploration
- Cited extensively in Related Work, Experimental Setup, and Results (Sections 2, 5, 6)

---

### [9] zitzler2003 - Performance Assessment of Multiobjective Optimizers: An Analysis and Review

**Citation**:
```bibtex
@article{zitzler2003,
  author = {Zitzler, Eckart and Thiele, Lothar and Laumanns, Marco and Fonseca, Carlos M. and da Fonseca, Viviane Grunert},
  title = {Performance assessment of multiobjective optimizers: an analysis and review},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = {2003},
  volume = {7},
  number = {2},
  pages = {117--132},
  doi = {10.1109/TEVC.2003.810758}
}
```

**Abstract**:
Comprehensive analysis of quality indicators for multi-objective optimization. Reviews properties of metrics including Pareto compliance, computational complexity, and ability to measure convergence vs diversity. Establishes theoretical foundations for comparing multi-objective algorithms.

**Relevance to PyCommend**:
- Justifies choice of Hypervolume, Spread, and ε-indicator as performance metrics
- Validates PyCommend's experimental methodology
- Cited in Experimental Setup (Section 5.4) for metrics selection

---

### [10] coello2007 - Evolutionary Algorithms for Solving Multi-Objective Problems

**Citation**:
```bibtex
@book{coello2007,
  author = {Coello Coello, Carlos A. and Lamont, Gary B. and van Veldhuizen, David A.},
  title = {Evolutionary Algorithms for Solving Multi-Objective Problems},
  edition = {2nd},
  year = {2007},
  publisher = {Springer},
  address = {New York},
  doi = {10.1007/978-0-387-36797-2}
}
```

**Abstract**:
Comprehensive textbook on evolutionary algorithms for multi-objective optimization. Covers fundamentals of Pareto optimality, various MOEA approaches (NSGA-II, MOEA/D, SPEA2, etc.), performance metrics, test problems, and applications. Emphasizes that population-based nature of EAs allows generation of multiple Pareto optimal solutions in single run, addressing complexities of multi-objective problems.

**Relevance to PyCommend**:
- Theoretical foundation for multi-objective formulation
- Justifies treating library recommendation as inherently multi-objective problem
- Supports discussion of competing objectives (LU vs SS vs RSS)
- Cited in Related Work (Section 2.2) for MOEA fundamentals

---

### [11] miettinen1999 - Nonlinear Multiobjective Optimization

**Citation**:
```bibtex
@book{miettinen1999,
  author = {Miettinen, Kaisa},
  title = {Nonlinear Multiobjective Optimization},
  year = {1999},
  publisher = {Kluwer Academic Publishers},
  address = {Boston},
  series = {International Series in Operations Research \& Management Science},
  volume = {12}
}
```

**Abstract**:
Comprehensive treatment of nonlinear multiobjective optimization theory and methods. Addresses problems with multiple conflicting or incommensurable objectives where traditional single-objective optimization and linear programming are insufficient. Provides extensive survey of state-of-the-art methods, theory, and mathematical foundations for deterministic multiobjective optimization including concepts of Pareto optimality, dominance, and trade-off analysis.

**Relevance to PyCommend**:
- Mathematical foundations for Pareto dominance in PyCommend
- Theoretical basis for multi-objective formulation with conflicting objectives
- Justifies non-weighted approach (Pareto front vs single weighted score)
- Cited in Problem Formulation (Section 3) for mathematical formulation

---

## 3. VARIABLE NEIGHBORHOOD SEARCH (3 references)

### [12] dahite2022 - Multi-Objective Model and Variable Neighborhood Search Algorithms

**Citation**:
```bibtex
@article{dahite2022,
  author = {Dahite, Lamiaa and Kadrani, Abdeslam and Benmansour, Rachid and Guibadj, Rym Nesrine and Fonlupt, Cyril},
  title = {Multi-Objective Model and Variable Neighborhood Search Algorithms for the Joint Maintenance Scheduling and Workforce Routing Problem},
  journal = {Mathematics},
  year = {2022},
  volume = {10},
  number = {11},
  pages = {1807},
  doi = {10.3390/math10111807},
  publisher = {MDPI}
}
```

**Abstract**:
Proposes MOVND/P and MOVND/PI algorithms using MOBI/P (Multi-Objective Best Improvement) Pareto local search strategy for maintenance scheduling problem. Demonstrates that VNS with Pareto-based local search achieves competitive results compared to NSGA-II and MOEA/D. Introduces systematic neighborhood exploration combined with elite archive management.

**Relevance to PyCommend**:
- **Core methodological foundation for MOVNS**
- MOBI/P strategy directly implemented in PyCommend's Pareto local search
- Validates VNS for multi-objective combinatorial optimization
- Cited extensively in Methodology (Section 4) for MOVNS algorithm design

---

### [13] arroyo2011 - Multi-objective Variable Neighborhood Search Algorithms

**Citation**:
```bibtex
@article{arroyo2011,
  author = {Arroyo, José Elias Claudio and Ottoni, Rafael dos Santos and Oliveira, Alcione de Paiva},
  title = {Multi-objective variable neighborhood search algorithms for a single machine scheduling problem with distinct due windows},
  journal = {Electronic Notes in Theoretical Computer Science},
  year = {2011},
  volume = {281},
  pages = {5--19},
  doi = {10.1016/j.entcs.2011.11.022}
}
```

**Abstract**:
Applies multi-objective VNS to scheduling problem with multiple due windows. Demonstrates effectiveness of VNS for combinatorial multi-objective problems. Uses Pareto archive to maintain non-dominated solutions and systematic neighborhood exploration for intensification and diversification.

**Relevance to PyCommend**:
- Validates VNS for discrete multi-objective optimization
- PyCommend's library recommendation is also combinatorial (subset selection)
- Supports neighborhood structure design
- Cited in Related Work (Section 2.3) for VNS applications

---

### [14] hansen2010 - Variable Neighbourhood Search: Methods and Applications

**Citation**:
```bibtex
@article{hansen2010,
  author = {Hansen, Pierre and Mladenović, Nenad and Pérez, José A. Moreno},
  title = {Variable neighbourhood search: methods and applications},
  journal = {Annals of Operations Research},
  year = {2010},
  volume = {175},
  number = {1},
  pages = {367--407},
  doi = {10.1007/s10479-009-0657-6}
}
```

**Abstract**:
Comprehensive survey of VNS methodology and applications. Describes fundamental principle of systematic neighborhood change for both intensification (local search) and diversification (shaking). Reviews applications across scheduling, routing, clustering, and other combinatorial optimization domains. Establishes VNS as general metaheuristic framework.

**Relevance to PyCommend**:
- Foundational reference for VNS methodology
- Justifies neighborhood structures (addition, removal, swap)
- Supports shaking mechanism design in MOVNS
- Cited in Related Work (Section 2.3) and Methodology (Section 4.2)

---

## 4. SEMANTIC SIMILARITY & EMBEDDINGS (2 references)

### [15] reimers2019 - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Citation**:
```bibtex
@inproceedings{reimers2019,
  author = {Reimers, Nils and Gurevych, Iryna},
  title = {Sentence-{BERT}: Sentence Embeddings using Siamese {BERT}-Networks},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
  pages = {3982--3992},
  doi = {10.18653/v1/D19-1410},
  publisher = {Association for Computational Linguistics}
}
```

**Abstract**:
Proposes Sentence-BERT (SBERT) using siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity. Achieves significant improvement over averaging BERT token embeddings, enabling efficient semantic similarity computation for large-scale applications.

**Relevance to PyCommend**:
- **Enables Semantic Similarity (SS) objective computation**
- PyCommend uses SBERT to generate 384-dimensional embeddings for 10,000 PyPI packages
- Cosine similarity between embeddings measures topical coherence
- Cited in Problem Formulation (Section 3) and Methodology (Section 4)

---

### [16] devlin2019 - BERT: Pre-training of Deep Bidirectional Transformers

**Citation**:
```bibtex
@inproceedings{devlin2019,
  author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
  year = {2019},
  pages = {4171--4186},
  doi = {10.18653/v1/N19-1423},
  publisher = {Association for Computational Linguistics}
}
```

**Abstract**:
Introduces BERT pre-training approach using masked language modeling and next sentence prediction on large text corpora. Bidirectional training enables deep understanding of language context. Achieved state-of-the-art results across eleven NLP tasks. Became foundation for numerous downstream applications including semantic similarity.

**Relevance to PyCommend**:
- Theoretical foundation for SBERT embeddings
- Enables deep semantic understanding of package descriptions
- Supports natural language processing of PyPI metadata
- Cited in Related Work (Section 2) for semantic similarity background

---

## 5. RECENT MOEA RESEARCH (1 reference)

### [17] liu2024 - Large Language Model Aided Multi-objective Evolutionary Algorithm

**Citation**:
```bibtex
@misc{liu2024,
  author = {Liu, Wanyi and Chen, Long and Tang, Zhenzhou},
  title = {Large Language Model Aided Multi-objective Evolutionary Algorithm: a Low-cost Adaptive Approach},
  year = {2024},
  eprint = {2410.02301},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE}
}
```

**Abstract**:
Proposes low-cost adaptive framework integrating Large Language Models with MOEAs (MOEA/D, NSGA-II, NSGA-III). LLMs provide adaptive guidance for algorithm parameter tuning and search strategy selection. Demonstrates that MOEA/D and NSGA-II remain standard baselines in 2024 research. Tests on ZDT and UF benchmark instances show improved performance with LLM integration.

**Relevance to PyCommend**:
- Validates choice of MOEA/D and NSGA-II as comparison baselines
- Demonstrates state-of-the-art 2024 research still uses these algorithms
- Potential future direction: LLM integration with MOVNS
- Cited in Related Work (Section 2.2) for recent MOEA developments

---

## 6. PERFORMANCE METRICS (2 references)

### [18] while2006 - A Faster Algorithm for Calculating Hypervolume

**Citation**:
```bibtex
@article{while2006,
  author = {While, Lyndon and Hingston, Philip and Barone, Luigi and Huband, Simon},
  title = {A faster algorithm for calculating hypervolume},
  journal = {IEEE Transactions on Evolutionary Computation},
  year = {2006},
  volume = {10},
  number = {1},
  pages = {29--38},
  doi = {10.1109/TEVC.2005.851275}
}
```

**Abstract**:
Presents efficient algorithm for calculating hypervolume indicator with improved computational complexity. Hypervolume measures both convergence and diversity of Pareto front approximation by calculating volume of objective space dominated by solution set. Widely adopted as primary quality indicator for multi-objective optimization.

**Relevance to PyCommend**:
- **Primary performance metric for PyCommend experiments**
- MOVNS achieves HV=0.748 vs NSGA-II HV=0.712 (+5.1%)
- Enables fair comparison between algorithms
- Cited in Experimental Setup (Section 5.4) and Results (Section 6)

---

### [19] zitzler2007 - The Hypervolume Indicator Revisited

**Citation**:
```bibtex
@article{zitzler2007,
  author = {Zitzler, Eckart and Brockhoff, Dimo and Thiele, Lothar},
  title = {The hypervolume indicator revisited: On the design of Pareto-compliant indicators via weighted integration},
  booktitle = {Evolutionary Multi-Criterion Optimization},
  year = {2007},
  pages = {862--876},
  doi = {10.1007/978-3-540-70928-2_64},
  publisher = {Springer}
}
```

**Abstract**:
Theoretical analysis of hypervolume indicator properties. Proves Pareto compliance and strict monotonicity with respect to Pareto dominance. Discusses design of quality indicators via weighted integration and relationship between hypervolume and other metrics. Establishes hypervolume as theoretically sound measure for multi-objective optimization.

**Relevance to PyCommend**:
- Theoretical justification for using HV as primary metric
- Validates experimental methodology
- Supports discussion of convergence and diversity trade-offs
- Cited in Experimental Setup (Section 5.4) for metric selection

---

## Summary Statistics

**By Category**:
- Software Library: 6 references (33.3%)
- Multi-objective EA: 5 references (27.8%)
- Variable Neighborhood Search: 3 references (16.7%)
- Semantic Similarity: 2 references (11.1%)
- Recent MOEA: 1 reference (5.6%)
- Performance Metrics: 2 references (11.1%)

**By Publication Type**:
- Journal Articles: 12 (66.7%)
- Conference Papers: 4 (22.2%)
- Books: 2 (11.1%)

**By Year**:
- 2024: 1 reference
- 2020-2022: 3 references
- 2010-2019: 7 references
- 2000-2009: 6 references
- Before 2000: 1 reference

**Key Venues**:
- IEEE Transactions on Evolutionary Computation: 4 papers
- Information and Software Technology: 2 papers
- Springer Books/Proceedings: 4 entries
- ACL NLP Conferences: 2 papers

---

## Citation Strategy for Paper Sections

### Section 1 (Introduction):
- Problem motivation: xu2020, auch2024
- Previous approaches: ouni2017, thung2013
- Search-based foundation: harman2001

### Section 2 (Related Work):
- Library recommendation: ouni2017, thung2013, xie2006
- Multi-objective EA: deb2002, zhang2007, coello2007, liu2024
- VNS methodology: hansen2010, dahite2022, arroyo2011
- Semantic similarity: reimers2019, devlin2019

### Section 3 (Problem Formulation):
- Multi-objective theory: miettinen1999, coello2007
- Data collection: xie2006 (repository mining precedent)
- Semantic embeddings: reimers2019

### Section 4 (Methodology):
- MOVNS design: dahite2022 (MOBI/P strategy)
- VNS fundamentals: hansen2010
- Pareto concepts: miettinen1999

### Section 5 (Experimental Setup):
- Baseline algorithms: deb2002 (NSGA-II), zhang2007 (MOEA/D)
- Performance metrics: zitzler2003, while2006, zitzler2007

### Section 6 (Results):
- Comparison with baselines: deb2002, zhang2007
- Hypervolume analysis: while2006, zitzler2007
- Real-world validation: xu2020 (library reuse patterns)

---

**End of Bibliography Documentation**
