# Top5 检索改进：七组受控实验

## 运行目标与比较方式

这组实验检验 PathCondRAG 的逐步改进是否提升 MuSiQue 的前 1、2、5 篇支撑文档召回。目标值 `Recall@5 ≈ 0.7528` 是实验目标，代码和本说明不保证达到该分数。由 `0.6528` 到 `0.7528` 是增加 **10 个百分点**，相对提高约 15.3%。

七组都重新检索，使用同一份原始 HippoRAG2 图和向量、同一批问题、同一个 Qwen3-Embedding-8B 与 qwen3-8b。实验仅执行检索，不执行 QA。`pathcondrag_original` 使用当前代码的 stage 0，并与其他 Path 组统一采用逐题 2/3/4 跳标签，因此不保证复现此前固定最大 2 跳的历史数值。

| 目录名称 | stage | 相对上一阶段增加的机制 |
|---|---:|---|
| `hipporag2` | — | 原始 HippoRAG2 检索对照 |
| `pathcondrag_original` | 0 | PathCondRAG 原有组件和 PC3 参数 |
| `exp1_correctness` | 1 | 实体真实文本、证据引用校验、QD 局部排名修复 |
| `exp2_evidence_candidates` | 2 | 恢复事实相关性，保留每条子问题候选的来源并融合 |
| `exp3_prefix_coverage` | 3 | 对前五篇做贪心证据覆盖选择与冗余惩罚 |
| `exp4_dependency_binding` | 4 | 显式子问题依赖 DAG，验证原文证据后绑定桥实体 |
| `exp5_verified_beam` | 5 | 最多保留三个一致的多桥实体分支，进行验证与 beam 搜索 |

stage 1—5 **逐步累加**。相邻两组差值表示该机制在已有机制上的增量作用，不能据此断言它独立使用时必然有效；交互作用需要后续单独消融。

## 固定配置

- 原始索引：`outputs/emb_ablation_qwen3emb8b_distinct_b4_retrieve/hipporag2_musique`。不从 PathCondRAG 的历史索引复制。
- 数据：`/root/datasets/musique.json`；语料：`/root/datasets/musique_corpus.json`，使用完整语料。
- embedding：`/root/models/Qwen3-Embedding-8B`，batch 固定为 4。
- LLM：`qwen3-8b`，默认 `http://127.0.0.1:8035/v1`。
- OpenIE 与检索 LLM workers 均为 8；进程内 HTTP 请求上限为 8。组间顺序运行。
- `max_new_tokens=2048`。Qwen 的客户端关闭 thinking；脚本不调整 vLLM 的模型长度或启用 thinking。
- Path 组使用 `hop_source=benchmark`、`hop_force_max=4`、`qd_max_sub_questions=4`。检索只获得逐题跳数；分解问题、中间答案、支撑文档标签不作为检索输入。
- 每题检索和导出 200 个候选；保留前 10 篇最终结果，用于 Recall@1/2/5/10/20/200 及错误诊断。
- 原 PC3 参数：迭代首轮 1 篇、第二轮种子 top5、`idf_novel`、迭代融合 0.45；QD 最小 2 跳、子检索 top3；PCQD evidence top5、entity top3、路径阈值 0.60、base/static/path 权重 0.40/0.20/0.40。

所有组的图和向量从原 Hippo 索引隔离复制。源 Hippo 的初始 `llm_cache` 冻结为单独快照，每组复制相同快照到自己的索引目录；某组新增的 LLM 缓存不会传给下一组。由于各机制的请求内容不同，实际请求数可能不同，报告保留这些开销。

## 启动命令

从项目根目录运行两题测试：

```bash
SAMPLE_SIZE=2 bash scripts/run_recall_top5_improvements.sh
```

两题默认选原始数据索引 `[0, 5]`，分别覆盖 2 跳和 4 跳。七组使用同一脚本、同一参数定义；小测试仅改变问题数量。全量运行：

```bash
SAMPLE_SIZE=0 bash scripts/run_recall_top5_improvements.sh
```

环境变量：

| 参数 | 默认与作用 |
|---|---|
| `OUT_ROOT` | 自动生成 `outputs/recall_top5_improvements_qwen3emb8b_b4_w8_<full\|smokeN>_<timestamp>` |
| `SOURCE_INDEX` | 上述原 Hippo 索引 |
| `SAMPLE_SIZE` | 0 表示全量；2 表示固定 2/4 跳测试；其他正值按 seed 抽样 |
| `SAMPLE_SEED` | 42 |
| `SAMPLE_INDICES_FILE` | 可选，JSON 整数列表；也接受含 `selected_indices` 的对象。原样保持顺序，不重新抽样 |
| `CASES` | 默认七组；可以传空格分隔的目录名称以选择子集 |
| `CONDA_ENV` | `rag` |
| `LLM_BASE_URL` | `http://127.0.0.1:8035/v1` |
| `VLLM_LOG` | `/root/eval/logs/vllm_qwen3.log`，要求可读取以核对状态码 |

显式传入 indices 时，正数 `SAMPLE_SIZE` 必须与索引数量一致；`SAMPLE_SIZE=0` 表示采用显式列表，不会扩展成全量。实际样本数和来源记录在 manifest 中。

脚本自动保存总日志 `logs/run.log` 以及每组日志。确认实际输出目录后：

```bash
tail -f <OUT_ROOT>/logs/run.log
```

## 输出与校验

```text
<OUT_ROOT>/
├── manifest.json                  # 数据/语料/索引身份、配置、样本与跳数
├── selected_indices.json          # 七组共用问题索引
├── initial_llm_cache/             # 源 Hippo 初始缓存快照
├── logs/run.log                   # 总日志
├── logs/<case>.log                # 每组日志
├── cases/<case>/
│   ├── index/                     # 隔离图、向量和独立缓存
│   ├── before.json                # 运行前哈希
│   ├── result.json                # 前10篇、前200候选、逐题指标与轨迹
│   ├── report.json                # 校验后的分层指标、耗时、请求及错误统计
│   └── validated.ok               # 成功校验标记及结果文件哈希
├── comparison.json               # 各组汇总与2/3/4跳分层
├── comparison.md                 # 主要指标表
└── completed.ok                  # 所有选定组都验证成功后生成
```

准备阶段验证全部 1000 题的跳数标签与 ID 一致、支撑文档在完整语料中、源索引 chunk 身份与完整语料完全一致。所有组共用一个问题清单。

每组结束后严格检查：

1. 样本 ID、原始索引、顺序、问题和跳数标签一致。
2. 每题有 10 篇唯一结果、200 篇唯一候选；最终结果与候选前缀一致，文档均来自语料。
3. 原图及三份 chunk/entity/fact embedding parquet 的 SHA256 在运行前后相同，源文件也保持相同。
4. batch、并发、模型、token 上限、stage 与预期一致。
5. 按每题支撑文档**完整字符串精确匹配**重新计算宏平均 Recall；逐题指标、全链命中和金文档 rank 与导出内容一致。
6. 对应时间段的 vLLM HTTP 状态没有非 200；请求失败计数为 0；实际 HTTP 调用应能在日志中找到状态记录。

失败即停止，不把失败组标记为成功，不自动删掉失败结果或静默降级。恢复时仅跳过已校验成功且哈希未改变的组。发现未完成的已有 case 目录会停止，保留现场供检查；需要重新运行时使用新的 `OUT_ROOT`。

## 阅读结果与论文定位

`Recall@5` 表示每题前五篇覆盖的支撑文档比例，再对问题做宏平均。`all_gold_top5` 表示前五篇包含全部支撑文档的题目比例，两者不能混用。同时检查前 1/2 篇表现、2/3/4 跳分层、候选 Recall@200 和 LLM 开销，区分候选缺失、排序错误和错误桥绑定。

已有工作包含：检索与推理交替的 [IRCoT](https://aclanthology.org/2023.acl-long.557/)、保留多条部分证据假设的 [Beam Retrieval](https://aclanthology.org/2024.naacl-long.96/)、按信息需求进行篇章集合选择的 [SetR](https://aclanthology.org/2025.acl-long.861/) 和使用图与三元组过滤的 [HippoRAG 2](https://arxiv.org/abs/2502.14802)。本实验检验这些思路在当前框架中的具体实现，不能将 beam、依赖分解或集合选择本身宣称为首创。

可研究的主线是：显式依赖与证据来源约束 → 桥实体候选绑定和验证 → 五篇预算内的证据覆盖。其新颖性和有效性需要进一步相关工作比较及严格消融。

当前七组不加载新 reranker。后续可用 [Qwen3 官方 reranker](https://github.com/QwenLM/Qwen3-Embedding) 作为排序瓶颈工程对照；若增加 reranker，HippoRAG2 也应采用相同候选、相同模型与相同预算，以免把辅助模型收益全部算作算法贡献。
