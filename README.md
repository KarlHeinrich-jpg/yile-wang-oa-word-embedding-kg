# 王毅乐开放论文语料的 Word Embedding 与知识图谱教学项目

本仓库把北京工商大学王毅乐参与发表的 6 篇开放获取英文论文整理成一套可以复现的教学案例。学生从已清洗的句子和词元开始，训练 Skip-gram Word2Vec，检查近邻词，构建论文—概念知识图谱，进行严格划分的链路预测，再完成文档相似度和语义检索。

仓库同时保留了原始来源的下载清单、许可信息、清洗后的训练数据、代码和一次可复现运行的结果。原始出版物属于各出版社或作者的开放许可；仓库中的代码和说明采用 MIT，数据使用条件以 `LICENSE-DATA.md` 和每篇文章的原始许可为准。

> **范围说明**：本仓库使用的是王毅乐相关的开放获取论文，不包含《Harry Potter》等受版权保护的小说全文。若要练习小说语料，请换用明确允许再分发的公共领域或开放许可文本。

## 你将学会什么

- 把 JATS XML 或 PDF 文章转成“一行一句、空格分词”的 `.txt` 语料，并保留句子级 JSONL 元数据。
- 理解中心词、上下文窗口、Skip-gram、负采样和余弦相似度。
- 训练并保存 Gensim Word2Vec，使用 `most_similar` 和词向量做检查。
- 把“文章提到概念”和“概念在句子中共现”记录成带来源的图谱边。
- 在训练图上隐藏一部分真实边，从真实非边中采样负例，用 Common Neighbors、Jaccard、Adamic–Adar 和逻辑回归比较链路预测。
- 用文章向量、TF–IDF 和查询词向量完成相似论文推荐，识别小语料中的常见词偏置。

## 快速运行

推荐 Python 3.11 或 3.12。CPU 上完整示例通常不到一分钟；第一次下载依赖的时间取决于网络。

```bash
git clone https://github.com/KarlHeinrich-jpg/yile-wang-oa-word-embedding-kg.git
cd yile-wang-oa-word-embedding-kg
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python code/run_all.py
```

`run_all.py` 按顺序运行 `01` 到 `05`：清洗（已有处理结果时会复用）、训练 Word2Vec、建图、链路预测、下游检索。重新从原始文件提取时运行：

```bash
python code/01_extract_and_clean.py --force
python code/run_all.py --skip-extract
```

运行完成后查看：

- `models/word2vec_skipgram_100d.txt`：标准 word2vec 文本格式，第一行是“词数 维度”。
- `graph/`：节点、边、GEXF 图和统计信息。
- `results/link_prediction_metrics.json`：AUC、Average Precision、测试集规模和划分校验。
- `results/predicted_links.csv`：候选概念对及结构分数。
- `results/document_similarity.csv`：论文相似度。
- `results/semantic_search.jsonl`：示例查询的检索结果。
- `figures/`：词向量 PCA、知识图谱和链路预测图。

### 当前基线运行结果

在 Python 3.12、Gensim 4.4.0、随机种子 42、单线程配置下，仓库随附结果为：

| 输出 | 结果 |
| --- | ---: |
| 句子数 | 1,283 |
| 词元数 | 34,254 |
| Word2Vec 词表 | 2,409 |
| 知识图谱节点 | 70（6 篇文章 + 64 个概念） |
| 知识图谱边 | 729（179 条提及、488 条共现、62 条语义相似） |
| 链路预测测试 ROC-AUC | 0.8549 |
| 链路预测测试 Average Precision | 0.8625 |

链路预测的 60%/20%/20% 边划分、负例来源和四项 `split_checks` 都保存在 `results/link_prediction_metrics.json`。重新训练或升级依赖后，浮点数可能发生小幅变化；报告结果时请以自己运行生成的 JSON 为准。

## 仓库结构

```text
.
├── README.md
├── LICENSE-CODE
├── LICENSE-DATA.md
├── requirements.txt
├── data
│   ├── raw
│   │   ├── SOURCES.md
│   │   └── source_manifest.json
│   └── processed
│       ├── corpus_sentences.jsonl
│       ├── corpus_sentences.txt
│       ├── corpus_tokenized.txt
│       ├── corpus_raw_clean.txt
│       ├── by_article/
│       └── token_frequencies.csv
├── models
├── graph
├── results
├── figures
├── code
│   ├── 00_download_sources.py
│   ├── 01_extract_and_clean.py
│   ├── 02_train_word2vec.py
│   ├── 03_build_knowledge_graph.py
│   ├── 04_link_prediction.py
│   ├── 05_downstream_recommendation.py
│   └── run_all.py
├── docs
│   └── tutorial.md
└── notebooks
    └── 01_word_embedding_walkthrough.ipynb
```

大文件模型和二进制缓存没有作为必需输入；代码会根据 `corpus_tokenized.txt` 重新生成它们。这样学生可以先阅读文本，再逐步理解二进制文件的作用。

## 数据来源与许可

| 编号 | 文章 | 年份 | 许可 | DOI 与来源 |
| --- | --- | ---: | --- | --- |
| 1 | *Analysis of Influencing Factors and Prediction of Provincial Energy Poverty in China Based on Explainable Deep Learning* | 2026 | CC BY 4.0 | [10.3390/systems14030319](https://doi.org/10.3390/systems14030319) |
| 2 | *Mapping the Coupling Coordination Between China’s Digital Economy and Carbon Emissions: Spatiotemporal Patterns and Spatial Markov Transitions* | 2026 | CC BY 4.0 | [10.3390/su18031283](https://doi.org/10.3390/su18031283) |
| 3 | *Research on photovoltaic power generation based on multi-dimensional indicators and models* | 2026 | CC BY 4.0 | [10.3389/fenvs.2026.1799258](https://doi.org/10.3389/fenvs.2026.1799258) |
| 4 | *Optimal Guarantee Level Optimization for Agricultural Insurance-Futures Based on CRRA Utility Maximization* | 2026 | CC BY 4.0 | [10.23977/agrfem.2026.090110](https://doi.org/10.23977/agrfem.2026.090110) |
| 5 | *Research on the Employment Prosperity Index of New Economic Service Industries in China based on Big Data* | 2025 | CC BY-NC 4.0 | [10.6981/FEM.202508_6(8).0022](https://doi.org/10.6981/FEM.202508_6(8).0022) |
| 6 | *A Hybrid PCA-Stacking Framework for Multidimensional Assessment of Development Trajectories: Evidence from China’s Modernization Process* | 2025 | CC BY-NC 4.0 | [10.6981/FEM.202507_6(7).0018](https://doi.org/10.6981/FEM.202507_6(7).0018) |

完整的作者、出版商页面、原始文件名和下载地址在 [`data/raw/SOURCES.md`](data/raw/SOURCES.md)。`data/raw/` 默认只提交来源清单，不把出版商 PDF 再打包进仓库；运行 `code/00_download_sources.py` 可以按清单下载。下载后的文件只用于许可允许的教学或研究用途，并请保留作者、DOI 和原许可声明。

## 数据处理约定

本案例的处理结果包括 6 篇文章、1,283 个句子和约 34,254 个词元。`corpus_sentences.jsonl` 每行保存一个句子，字段有 `doc_id`、标题、DOI、年份、许可、句子编号、原文和 `tokens`。`corpus_tokenized.txt` 每行是对应的空格分隔小写词元，适合直接传给 Gensim 的 `corpus_file`。

清洗规则在 `code/01_extract_and_clean.py` 中实现：

1. XML 读取标题、摘要和正文段落；PDF 读取页面文本，并在 References 之前停止。
2. 规范 Unicode 空白，去掉重复的页眉、页码、出版商提示和明显重复句子。
3. 用句末标点切句；保留常见缩写中的句点。
4. 使用正则表达式保留英文词、连字符/撇号词和数字；全部转小写。
5. 把句子级元数据写入 JSONL，便于回溯某个词来自哪篇文章。

这套规则适合课堂演示，不等同于经过语言学专家审核的出版级语料。正式研究应检查公式、表格、参考文献和 PDF 双栏顺序，并保留原始文件的 SHA-256。

## Word2Vec 配置

默认模型是 Skip-gram，参数固定在 `code/02_train_word2vec.py`：

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `vector_size` | 100 | 每个词的向量维度 |
| `window` | 5 | 左右最多各看 5 个词 |
| `min_count` | 2 | 低于两次的词不进词表 |
| `sg` | 1 | Skip-gram；设为 0 可改成 CBOW |
| `negative` | 10 | 每个正样本抽取的噪声词数量 |
| `sample` | 1e-4 | 高频词下采样阈值 |
| `epochs` | 20 | 训练轮数 |
| `workers` | 1 | 为了课堂复现稳定；生产环境可增加并注明结果会有微小抖动 |
| `seed` | 42 | 随机初始化种子 |

Skip-gram 的训练目标可以写成

$$\max \sum_{(w,c)\in D}\log \sigma(v_c^\top u_w)+\sum_{i=1}^{k}\mathbb{E}_{n_i\sim P_n}\log \sigma(-v_{n_i}^\top u_w).$$

式中 `u` 和 `v` 是中心词与上下文向量，`D` 是窗口产生的正样本，`k` 是负采样数。训练结束后通常使用 `model.wv` 查询向量；模型状态和只读向量的保存方式不同，详见教程中的说明。

## 知识图谱与链路预测

`03_build_knowledge_graph.py` 建立三类节点：文章节点、人工审核的教学概念节点，以及由句子共现得到的概念节点。边带有 `relation`、`weight`、`source_sentence_ids` 等字段，具体关系包括：

- `mentions`：文章在多少个句子中提到概念；
- `co_occurs`：两个概念出现在同一句子的次数；
- `semantic_similar`：Word2Vec 余弦相似度达到阈值的概念对。

`04_link_prediction.py` 采用结构性评估：先从真实图中固定随机种子抽出一部分正边作为测试集，训练图中不含这些边；负例只从原始图真正不存在的节点对中抽取，且与正例和训练边互斥。所有特征只由训练图计算，避免把待预测的边泄露给模型。指标是 ROC-AUC 和 Average Precision；候选分数用于排序，不能直接解释成现实世界发生概率。

这个任务预测的是“论文概念关系是否值得进一步核查”，不是经济指标、政策效果或因果关系的预测。六篇文章组成的语料很小，AUC 会随随机种子和概念词表变化，课堂中应报告划分方式和局限，而不是只展示一个高分。

## 下游任务

`05_downstream_recommendation.py` 有两个可拆开的练习：

1. 把文章中词向量做均值，计算论文之间的余弦相似度；同时给出 TF–IDF 结果作对照。
2. 输入 `energy poverty`、`digital carbon` 等查询词，返回最相近的句子和文章。

平均向量在小语料里容易被 `the`、`model`、`development` 等通用词主导。学生可以尝试停用词、TF–IDF 加权平均、短语检测或更大的公开语料，并比较结果是否更贴近人工判断。

## 课程建议

- 第 1 次课：只运行 `01`、统计词频，手工检查 20 条 JSONL。
- 第 2 次课：改变 `window`、`min_count`、`vector_size`，记录词表和近邻变化。
- 第 3 次课：从一句话画出窗口产生的正样本，解释负采样。
- 第 4 次课：检查图谱边的来源句子，比较共现权重和语义相似边。
- 第 5 次课：删去 `04` 中一个特征，观察 AUC 和 AP 是否变化；说明数据泄露。
- 第 6 次课：把概念词表从人工名单改为 TF–IDF 或关键词抽取，写一页误差分析。

更长的教学讲义、代码解释、常见错误和扩展方向在 [`docs/tutorial.md`](docs/tutorial.md)。

## 参考文档

- [Gensim Word2Vec API](https://radimrehurek.com/gensim/models/word2vec.html)
- [scikit-learn 常见数据泄露问题](https://scikit-learn.org/stable/common_pitfalls.html)
- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [NetworkX link prediction documentation](https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html)

## 贡献与引用

欢迎提交 issue，说明数据清洗错误、可复现问题或教学改进。引用论文时请使用各自 DOI；引用本案例时可注明：Yile Wang, *Word Embedding and Knowledge Graph Teaching Project*, 2026, GitHub repository.
