# exp4 + plan_prune + dag_package 全量检索

## 运行

```bash
cd /root/PathCondRAG
bash scripts/run_multi_dataset_retrieval_plan_prune_dag_package.sh
```

同一个脚本支持两题检查：

```bash
RUN_LOG=outputs/3multi_hop_datasets_results_10_5/_plan_prune_dag_package_smoke/launcher.log \
  bash scripts/run_multi_dataset_retrieval_plan_prune_dag_package.sh --smoke
```

检查结果位于 `_plan_prune_dag_package_smoke`，与全量结果隔离。
检查通过后可以删除该目录；保留旧的公共索引和已有实验结果。

## 配置

| 项目 | 设置 |
|---|---|
| 数据集 | HotpotQA、2WikiMultiHopQA、MuSiQue，各完整 1000 题 |
| 模块 | `planning,plan_prune,dag_package` |
| 规划校验、路由 | `canonical_refs`、`question_structure` |
| 索引 | 复用 `shared_indexes/<dataset>`，图和向量内容不变 |
| embedding | Qwen3-Embedding-8B，batch=4 |
| LLM | qwen3-8b，检索并发 8，索引并发参数 8，thinking 关闭 |
| 输出长度 | 2048，不修改 |
| 跳数 | benchmark；MuSiQue 按题读取，HotpotQA/2Wiki 使用现有数据集先验 |
| 评估 | 仅检索，保存逐题 Top10、Top200 候选和追踪信息 |

结果：

```text
outputs/3multi_hop_datasets_results_10_5/
  cases/<dataset>/exp4_dependency_binding_plan_prune_dag_package/
    result.json
    report.json
    dag_validation.json
    comparison_vs_plan_prune.json
    run.log
  logs/run_plan_prune_dag_package_full.log
  metadata/exp4_dependency_binding_plan_prune_dag_package_comparison.json
  metadata/exp4_dependency_binding_plan_prune_dag_package_comparison.md
```

运行按 HotpotQA → 2Wiki → MuSiQue 顺序进行。已验证的完整结果可以跳过；
未完成目录不会被脚本自动覆盖。验证后清理每组私有索引，保留检索结果。

## 比较说明

初始 LLM 缓存采用原 exp4 的私有快照，与现有全量 plan_prune 脚本一致。
之前全量 plan_prune 的完成缓存已清理，因此其保存的结果属于历史对照。
`comparison_vs_plan_prune.json` 记录相同问题的指标差值和上游规划、绑定、
LLM 输出是否一致；存在差异时，不把全部增幅都归因于 DAG 选择。

`dag_validation.json` 检查 DAG finalizer 已启用、未读取 gold、没有新增直接
LLM 请求，并保留其自身父排序的 Top2 和完整文档集合。
