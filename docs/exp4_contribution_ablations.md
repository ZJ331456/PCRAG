# exp4 的四类贡献验证

统一使用此次重建的 HippoRAG2 索引，Qwen3-Embedding-8B batch=4、qwen3-8b LLM 并发上限8、max_new_tokens=2048、thinking=false。检索输入保留逐题 benchmark 跳数；不执行 QA。四类比较展开为11个运行组。

| 类别 | 组名后缀 | 控制内容 |
|---|---|---|
| exp4_abla1 | budget_dag / budget_qd / budget_iterative | 同一逐题额外证据模块调用上限；DAG、独立子问题扩展、上下文驱动多轮查询 |
| exp4_abla2 | fixed_coverage / fixed_binding | 冻结exp3每题Top200，所有证据模块检索只在该池打分；最终200篇集合完全相同 |
| exp4_abla3 | string / literal / relation | 答案字符串、答案与原文引用核验、额外独立关系支持判断 |
| exp4_abla4 | coverage / ancestor / joint | 统一分支搜索；普通覆盖、保留祖先支撑、在生成的绑定分支与证据集合之间做有限联合搜索 |

## 实验边界

预算是原exp4逐题 `llm_plan_calls + llm_verification_calls`，包含语义修正。共同的事实过滤、静态QD、PCQD流程保持相同。QD与迭代对照将每次额度用于生成/改进查询；DAG可能提前停止，所以报告额度、实际逻辑调用、HTTP请求分别列出。本实验不是等token或等耗时，不能据此宣称控制了所有计算成本。缓存命中仍计入逻辑调用，网络重试单独计数。

固定池仅向检索器提供每篇chunk哈希和既有分数，以及原问题和预算数字；不传历史gold、分解内容、历史绑定答案或历史规划。每个新组仍从真实索引读取正文和向量。固定池组的Recall@200应一致，其Top5变化隔离候选覆盖差异。

字符串与字面引用组使用相同的提取提示词，只改变接纳答案时的引用检查。字面引用核验只检查来源和答案出现，不保证关系蕴含。relation组通过同一个LLM的另一项关系判断任务核对实体、关系、类型与限定条件，仍是模型判断，不是逻辑正确性保证。它增加请求成本，必须同时阅读成本列。

第四类三组均使用相同宽度的有限分支搜索，避免仅joint组新增候选。coverage/ancestor使用局部最佳绑定分支；joint比较生成的分支及祖先闭合的proof种子集，再贪心填充和有限交换。联合优化范围受生成的分支与候选限制，不保证所有可能绑定的全局最优。这两项新选择策略属于增强机制验证，并非仅删除已有组件。

所有新组使用独立元数据和SQLite缓存；图与三份向量文件以硬链接复用且禁止重建，前后核对SHA。节省磁盘不改变算法输入。第四类coverage完成后冻结此次共同探索的响应缓存，ancestor/joint复制该快照回放；报告强制检查逐题计划、路线、分支证明和逻辑调用的指纹一致。共同探索成本由coverage组记录，后两组缓存回放的HTTP减少不能解释为其算法更高效。原七组清单、comparison与completed标记保留；新清单为ablation_manifest.json，汇总为ablation_comparison.json/md。

新增证据模块对相同提示词协调请求：首次响应写入缓存后，后续逻辑调用读取相同响应，避免并发覆盖导致分支无法回放；不同提示词仍受原来的8并发上限调度。每次逻辑调用仍计入预算及逻辑token成本。

## 运行

两题（2跳与4跳）的完整11组端到端测试：

```bash
SAMPLE_SIZE=2 bash scripts/run_exp4_contribution_ablations.sh
```

全量1000题：

```bash
SAMPLE_SIZE=0 bash scripts/run_exp4_contribution_ablations.sh
tail -f /root/PathCondRAG/outputs/pathcondrag_new_innvotion_10_1/logs/run2.log
```

小测试默认使用独立outputs/exp4_contribution_smoke2_时间目录。原索引不重建；每题保存Top10、200候选和完整诊断。任何索引变更、固定池变更、预算超限、非200或请求失败都会停止后续组。

## 本次验证

2026-10-02：45项针对预算、并发、固定候选、绑定门禁、祖先闭合和回放完整性的单元测试通过。同一完整脚本在 `outputs/exp4_contribution_smoke2_20261002_v1` 跑完11组、每组2题（2跳与4跳），全部验证通过；合计100次HTTP请求，失败0、非200响应0。第四类三组的探索轨迹指纹一致，图与向量SHA保持一致。两题只用于检查实验流程，不用于判断性能收益。
