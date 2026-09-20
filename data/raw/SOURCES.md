# 原始来源与下载说明

本目录的 `source_manifest.json` 是机器可读的来源清单。出于仓库体积、许可边界和可复现性考虑，原始 PDF/XML 默认不提交；执行 `python code/00_download_sources.py` 可下载到本目录，再运行 `python code/01_extract_and_clean.py --force` 重新抽取。下载器不会跳过许可检查：学生仍需按文章的原始许可使用文件。

| 文档 ID | 原始格式 | 许可 | 出版商页面 | 下载入口 |
| --- | --- | --- | --- | --- |
| `wang_2026_energy_poverty` | JATS XML | CC BY 4.0 | [Analysis of Influencing Factors and Prediction of Provincial Energy Poverty in China Based on Explainable Deep Learning](https://www.mdpi.com/2079-8954/14/3/319) | [XML](https://www.mdpi.com/2079-8954/14/3/319/xml) / [PDF](https://www.mdpi.com/2079-8954/14/3/319/pdf) |
| `wang_2026_digital_carbon` | JATS XML | CC BY 4.0 | [Mapping the Coupling Coordination Between China’s Digital Economy and Carbon Emissions: Spatiotemporal Patterns and Spatial Markov Transitions](https://www.mdpi.com/2071-1050/18/3/1283) | [XML](https://www.mdpi.com/2071-1050/18/3/1283/xml) / [PDF](https://www.mdpi.com/2071-1050/18/3/1283/pdf) |
| `wang_2026_photovoltaic` | JATS XML | CC BY 4.0 | [Research on photovoltaic power generation based on multi-dimensional indicators and models](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1799258/full) | [XML](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1799258/xml/nlm) / [PDF](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1799258/pdf) |
| `wang_2026_insurance_futures` | PDF | CC BY 4.0 | [Optimal Guarantee Level Optimization for Agricultural Insurance-Futures Based on CRRA Utility Maximization](https://www.clausiuspress.com/article/18020.html) | [PDF](https://www.clausiuspress.com/assets/default/article/2026/09/14/article_1789382332.pdf) |
| `wang_2025_employment_index` | PDF | CC BY-NC 4.0 | [Research on the Employment Prosperity Index of New Economic Service Industries in China based on Big Data](https://fieam.org/index.php/ojs/article/view/74) | [PDF](https://fieam.org/index.php/ojs/article/download/74/78/152) |
| `wang_2025_hybrid_pca_stacking` | PDF | CC BY-NC 4.0 | [A Hybrid PCA-Stacking Framework for Multidimensional Assessment of Development Trajectories: Evidence from China’s Modernization Process](https://fieam.org/index.php/ojs/article/view/49) | [PDF](https://fieam.org/index.php/ojs/article/download/49/53/101) |

## 署名模板

处理数据或输出图表时，请同时写出作者、题名、DOI、出版商 URL 和原始许可。例如：

> Fan, Zihao; Fan, Pengying; Wang, Yile. “Analysis of Influencing Factors and Prediction of Provincial Energy Poverty in China Based on Explainable Deep Learning.” DOI: 10.3390/systems14030319. CC BY 4.0.

四篇 CC BY 4.0 和两篇 CC BY-NC 4.0 的适用范围不同；详细说明见根目录 `LICENSE-DATA.md`。
