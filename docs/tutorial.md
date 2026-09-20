# Word Embedding 与知识图谱实验教程

## 1 实验目标和数据边界

本实验把六篇开放获取英文论文当作一个很小的领域语料。它适合讲清楚数据结构、算法接口和评估方法，不能代表“英语语言”或“中国经济研究”的完整分布。学生应把每个结果理解为这六篇文章中的统计关系，不能直接把相似度解释成同义关系，更不能把链路预测分数当成政策或经济预测。

完成实验后，你应该能够回答四个问题：

1. 一行一句、空格分隔的文本为什么适合传给 Word2Vec？
2. 词向量的近邻是由什么上下文证据产生的？
3. 知识图谱中的一条边如何回溯到原始句子？
4. 如何设计一个不会把测试边泄露给模型的链路预测实验？

项目使用 `data/processed/corpus_sentences.jsonl` 作为可审计的中间层。每一行包含文章 ID、题名、DOI、许可、句子编号、原文和词元列表。`corpus_tokenized.txt` 只保留每行的词元，适合作为训练输入。原始文件下载方式和许可见 `data/raw/SOURCES.md` 与 `LICENSE-DATA.md`。

## 2 环境与第一次运行

建议使用 Python 3.11 或 3.12。建立虚拟环境后安装固定的大版本依赖：

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

先直接运行现成的派生数据：

```bash
python code/run_all.py --skip-extract
```

如果希望从出版商文件重新走一遍抽取流程：

```bash
python code/00_download_sources.py
python code/01_extract_and_clean.py --force
python code/run_all.py --skip-extract
```

下载脚本使用清单中的 URL 和 User-Agent，并把下载文件的 SHA-256 写入 `data/raw/download_manifest.json`。如果某个站点暂时拒绝请求，不要反复重试；保留错误信息，手动从出版商页面下载后放到 `data/raw/`，再运行提取脚本。

## 3 从原始文章到训练语料

### 3.1 为什么需要中间层

直接把 PDF 交给算法会混入页眉、页码、参考文献、表格列顺序和出版商提示。`01_extract_and_clean.py` 把读取、切句、分词和元数据保存分开，任何一条结果都可以回到 `doc_id` 和 `sentence_id`。这种可追溯性比把清洗逻辑藏在一个 Notebook 单元里更适合教学和研究复核。

### 3.2 XML 与 PDF 的读取

对 JATS XML，脚本读取 `article-title`、`abstract/p`、`body/p` 和 `body/title`。对 PDF，脚本逐页读取文本，在 `References` 或 `Bibliography` 之前停止，并删除反复出现的短行和纯页码。PDF 双栏、公式和表格仍可能需要人工检查；脚本的目标是给出清楚的基线，而不是替代版面分析器。

核心函数的结构如下：

```python
from lxml import etree
import fitz

xml = etree.parse("data/raw/wang_2026_energy_poverty.xml")
paragraphs = xml.xpath("//abstract//p | //body//p")
text = "\\n".join(" ".join(p.itertext()) for p in paragraphs)

pdf = fitz.open("data/raw/wang_2025_employment_index.pdf")
pdf_text = "\\n".join(page.get_text("text") for page in pdf)
```

### 3.3 切句与分词

示例分词器使用下列正则表达式：

```python
TOKEN_RE = r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\\d+(?:\\.\\d+)?"
```

它会保留 `risk-based`、`farmer's` 和 `2.5`，再统一转成小写。它不会做词形还原，也不会自动把 `photovoltaic power` 变成一个词。这样的取舍让学生能直接看到每一步发生了什么。

检查数据时，不要只看总行数。建议抽查：

```python
import json
from pathlib import Path

path = Path("data/processed/corpus_sentences.jsonl")
for line in path.open(encoding="utf-8"):
    row = json.loads(line)
    if row["doc_id"] == "wang_2026_energy_poverty" and row["sentence_id"] < 3:
        print(row["sentence_id"], row["text"])
        print(row["tokens"])
```

再检查每篇文章的句子数、词元数和高频词。若出现大量 `figure`、邮箱域名或页眉短语，先修正清洗规则，再训练模型。

## 4 Word2Vec 的输入、目标与输出

### 4.1 输入格式

`corpus_tokenized.txt` 每行一个句子，词元之间只有空格：

```text
energy poverty refers to a condition in which individuals cannot access essential services
photovoltaic power faces challenges related to weather stability and construction costs
```

Gensim 的 `corpus_file` 会逐行流式读取，不必先把所有句子放进内存。文件可读、可复用，也方便学生用 `head` 或 Python 观察训练样本。

### 4.2 窗口如何生成正样本

假设句子是 `digital economy supports carbon reduction`，窗口为 2，则中心词 `economy` 产生的正上下文包括 `digital`、`supports` 和（按边界）`carbon`。窗口越大，模型看到的主题关系越宽；窗口越小，更偏向局部搭配。

Skip-gram 让中心词预测上下文。负采样把每个真实的 `(center, context)` 对与若干噪声词进行二分类。可写成：

$$\log \sigma(v_c^\top u_w)+\sum_{i=1}^{k}\log \sigma(-v_{n_i}^\top u_w).$$

`negative=10` 意味着每个正样本平均抽取十个噪声词。它使用独立的二分类训练目标，区别于完整 softmax，目标是得到可比较的向量，而不是提供一个可解释的概率模型。

### 4.3 运行训练代码

```bash
python code/02_train_word2vec.py
```

代码使用 `workers=1` 和稳定哈希函数，减少多线程调度造成的结果抖动。实际项目可把 `workers` 调大来加速，但要在报告中说明 Gensim 文档所说的非完全确定性。训练后有三种常见输出：

```python
from gensim.models import Word2Vec, KeyedVectors

model = Word2Vec.load("models/word2vec_skipgram_100d.model")
print(model.wv["energy"].shape)              # (100,)
print(model.wv.most_similar("energy", topn=10))

# 只分发向量时可加载 C word2vec 文本格式
wv = KeyedVectors.load_word2vec_format("models/word2vec_skipgram_100d.txt", binary=False)
```

完整 `.model` 保存了继续训练所需的状态；`.txt` 只保存词和向量，便于跨语言工具读取；`.bin` 是兼容原始 word2vec 的二进制格式。三者不是三套不同模型。

### 4.4 近邻结果应该怎样读

`most_similar("energy")` 返回的是训练词表中的余弦相似度排序。它更接近“在当前语料中共享上下文的词”，而不是词典意义上的同义词。小语料会产生三个常见现象：

- 论文写作模板词靠得很近；
- 同一个缩写或拼写变体被分开；
- 低频词的方向不稳定。

学生可以把 `window` 改为 2、8，把 `min_count` 改为 1、5，记录词表大小和几个固定查询的近邻变化，并写出一段解释。

## 5 从句子到知识图谱

### 5.1 节点和关系

`03_build_knowledge_graph.py` 采用一个显式的课堂概念表。节点分为：

- `article:...`：六篇文章；
- `term:...`：能源、碳排放、光伏、保险、就业等概念。

边有三种：

- `mentions`：某篇文章提到概念的句子数；
- `co_occurs`：两个概念在同一句子中同时出现的次数；
- `semantic_similar`：Word2Vec 近邻且余弦相似度达到阈值。

概念表是教学选择，不是自动实体识别。文章—概念边和共现边都保存 `source_sentence_ids`，因此可以从 `graph/edges.csv` 回到 JSONL 检查证据。

### 5.2 共现统计的最小代码

```python
from itertools import combinations
from collections import Counter

present = sorted(set(tokens) & CONCEPTS)
for left, right in combinations(present, 2):
    cooccurrence[(left, right)] += 1
```

脚本只把至少出现两次的共现对写入图谱，以免单个偶然句子占据图面。把 `cooccurrence_min_count` 改成 1 可以观察图的稠密化。

### 5.3 读取和检查图

```python
import networkx as nx

g = nx.read_gexf("graph/knowledge_graph.gexf")
print(g.number_of_nodes(), g.number_of_edges())
for u, v, data in g.edges(data=True):
    if data.get("relation") == "co_occurs":
        print(u, v, data["weight"], data["source_sentence_ids"])
        break
```

`concept_graph.gexf` 只保留概念节点，适合链路预测和画图；`knowledge_graph.gexf` 保留文章节点和出处。图上的边表示语料中的观察关系，不能直接解释成现实世界的因果边。

## 6 链路预测的严格划分

### 6.1 为什么容易泄露

一个常见错误是先把完整图交给 `nx.non_edges` 或先计算包含全部边的节点度，再把一部分边称为测试集。这样测试边或其邻接信息已经进入特征，AUC 会被高估。另一个错误是把 Word2Vec 在全部文本上的相似度当作“未见信息”。这在探索性分析里可以接受，但必须标记为传导式实验。

本项目把结构性评估单独出来：

1. 取所有真实 `co_occurs` 边，随机打乱后按 60%/20%/20% 分为训练、验证、测试正例。
2. 负例只从完整观察图的真正非边抽取，三个负例集合互不相交。
3. 建立只含训练正边的 `base` 图。
4. 对每个节点对计算 Common Neighbors、Jaccard、Adamic–Adar 和 Preferential Attachment；验证和测试特征都只看 `base`。
5. 训练正例先临时删去自身边再计算特征，用训练正负例拟合逻辑回归，在测试集报告 ROC-AUC 和 Average Precision。

### 6.2 运行和解释结果

```bash
python code/04_link_prediction.py
cat results/link_prediction_metrics.json
```

`split_checks` 必须全部为 `true`。如果出现 `false`，先修复实验，再讨论指标。AUC 衡量正例排在负例前面的概率，AP 更重视排序靠前的正例。这里每个集合正负数量相同，所以它们适合比较不同代码版本；它们仍然不是全图上每一对概念的发生概率。

脚本最后用所有已观察共现边训练一个用于候选排序的模型，结果写到 `results/predicted_links.csv`。候选边需要人工回看其共同邻居、相关句子和论文来源。课程报告应写清楚“这是待核查的关系候选”，不要写成“模型发现了真实关系”。

## 7 文档向量和语义检索

### 7.1 均值向量

给定一篇文章的词元集合 `T`，简单文档向量是：

$$d=\frac{1}{|T|}\sum_{w\in T}u_w.$$ 

`05_downstream_recommendation.py` 对六篇文章都计算均值向量，并与 TF–IDF 余弦相似度并列输出。均值方法容易被常见词影响；TF–IDF 会降低只在很多文章出现的词的权重。

### 7.2 查询句子

脚本对五个查询分别求词向量均值，再与每个句子的均值向量计算余弦相似度：

```bash
python code/05_downstream_recommendation.py
head -n 2 results/semantic_search.jsonl
```

查询词不在词表时会被跳过；如果一个查询没有任何已知词，结果不应被解释。学生可以把 `mean_vector` 改成 TF–IDF 加权平均，或者先用 `gensim.models.Phrases` 把 `energy_poverty` 作为短语再训练。

## 8 图和结果的可读性

```bash
python code/06_make_figures.py
```

PCA 图只用于二维展示，不代表原空间中只有两个重要维度。概念图只画权重最大的 25 条边，以免 64 个节点全部标注导致不可读。链路预测图中的横轴是排序分数，不是概率校准值。

查看输出时同时检查数值和图形：图的标签是否重叠、CSV 是否有重复节点对、JSONL 是否能被逐行解析。一个看起来漂亮但没有来源句子的图，不具备教学上的可追溯性。

## 9 可复现实验记录

每次改参数都应记录：

```text
日期：
Python / gensim / networkx 版本：
语料文件 SHA-256：
vector_size / window / min_count / sg / negative / epochs：
随机种子与 workers：
概念表版本：
图谱共现阈值：
链路预测划分比例：
ROC-AUC / AP：
人工检查的错误案例：
```

如果要发表结果，建议把完整环境写入 `requirements-lock.txt`，并在论文中说明语料只包含六篇文章、哪些是作者自有内容、哪些数据受 CC BY-NC 的非商业条件约束。

## 10 练习与扩展

1. **词表敏感性**：把 `min_count` 分别设为 1、2、5，比较低频概念是否进入图谱。
2. **停用词实验**：只在文档均值阶段去掉停用词，保持 Word2Vec 训练输入不变，比较推荐结果。
3. **时间切分**：把 2025 文章当训练集、2026 文章当测试集，讨论两篇文章是否足够支持时间外推。
4. **自动概念表**：用 TF–IDF 前 80 个词或关键词抽取替代人工 `CONCEPTS`，再人工审核 20 个候选。
5. **关系类型**：从句法模板抽取 `predicts`、`reduces`、`supports` 等有方向关系，比较它们和共现边的差异。
6. **知识图谱嵌入**：在本图上尝试 TransE 或 RotatE，说明只有三类关系和很少节点时为什么容易过拟合。
7. **更大公开语料**：加入许可清楚的政策文件或论文摘要，并报告数据规模变化，而不是把模型结果直接与本案例比较。

每个扩展都要保留一个简单基线和错误分析。课堂学习的重点是把数据、假设、代码和评估连起来，而不是追求一个孤立的最高分。
