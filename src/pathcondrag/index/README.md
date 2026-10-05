# 索引构建代码

## 目录

| 位置 | 职责 |
| --- | --- |
| `ner/` | 在线与离线命名实体抽取、NER 校验和重试 |
| `openie/` | 三元组抽取、本地结构校验；可选原文证据核验和恢复 |
| `shared_index_builder.py` | 共享索引构建入口及 HippoRAG 运行时适配 |
| `openie_build_queue.py` | 协调 NER 和三元组阶段、控制并发、恢复未完成片段 |
| `openie_checkpoint.py` | 保存和读取逐阶段 SQLite 检查点 |
| `extraction_utils.py` | 两阶段共用的响应解析、环境变量和重试设置 |

在线索引默认使用 `StructuralOpenIE`：组合 NER 与三元组阶段，对抽取结果做
本地结构校验。`SourceVerifiedOpenIE` 是独立可选的高成本模式，会额外请求 LLM
核验关系。两种模式均保持 NER 输出上限 512、三元组输出上限 2048。
离线后端共用批量 NER 与 OpenIE 实现。

`scripts/build_shared_index.py` 仍是命令行入口。
信息抽取实现统一位于 `ner/` 与 `openie/`，项目内部均引用这些新路径。
离线模型依赖只在构造相应后端时加载。

## 失败策略与提示词

构建入口可以追加以下参数，`eval_dataset.py` 和新索引对比脚本也支持它们：

```bash
--openie_validation_mode structural --openie_strict true --openie_prompt_version optimized
```

- `openie_validation_mode=structural`（默认）：要求三元组恰有三个合法字符串字段、
  返回正常结束等本地条件。非空原文最终仍没有合法关系时记录抽取失败，不自动批准
  为空；只含空白的原文可由本地检查确认为空。**结构合法不代表关系语义正确。**
- `openie_validation_mode=source_verified`：额外执行独立原文证据与角色核验；失败时
  可以进入 compact、atomic 等恢复策略。适用于需要额外核验且接受较高成本的构建。
- `openie_strict=true`（默认）：要求所有 chunk 满足所选模式的完成合同，失败则
  保存诊断并中止图发布；它不自动切换为 `source_verified`，也不保证 LLM 无语义错误。
- `openie_strict=false`：所选模式的抽取和重试规则保持一致，最终失败会警告并记录统计，
  对应 chunk 的文本和向量保留，未通过检查的关系不入图，继续完成索引。
  该选项只放宽抽取失败；语料身份冲突、损坏检查点等错误仍会中止。
- `openie_prompt_version=origin`：使用此次优化前保存的
  `prompts/templates/origin_ner_prompt.py` 和 `origin_triple_extraction_prompt.py`。
- `openie_prompt_version=optimized`（默认）：使用优化后的 `ner.py` 和
  `triple_extraction.py`。`origin` 指本项目优化前版本，不是承诺与原生 HippoRAG
  提示词完全相同。

验证模式与失败发布策略相互独立。模式、失败策略、所选提示词和提示词哈希写入质量身份。
变更后应在独立目录重建索引，或执行有备份、有记录的显式检查点迁移，
不把旧检查点误当作新提示词的实验结果。原始 HippoRAG 源码不作修改。

## 开销

NER 和三元组各阶段最多 8 个 worker，保持 NER 512、三元组 2048、Qwen3 thinking 关闭。
默认结构模式下，普通成功 chunk 通常需要 **NER 1 次、三元组 1 次**逻辑请求；
不会调用语义 audits、compact 或 atomic 恢复。只有格式错误、截断等失败才重试，
每个 chunk 在当前结构 profile 下的**三元组逻辑 infer 累计上限为 3 次**。
此预算不含 NER 请求和 API 客户端底层 HTTP 重试；迁移前已发生的旧请求另行保留历史统计。
队列重新调度和检查点续跑不能重置当前结构 profile 的三元组预算。

预检查与实际请求复用精确 token 计数和语法编译缓存。已完成的阶段由检查点复用。
失败 NER 原文片段会继续按原始边界细分，成功片段可恢复复用；最多 64 次累计
NER 请求、12 层细分，只有覆盖全部原文时才标记完整。恢复反馈带尝试编号，避免
相同失败反馈一直命中同一个失败响应缓存。固定长度数组通过等价 `prefixItems`
编译，绕开当前 xgrammar 对 `minItems/maxItems` 的忽略。

`source_verified` 会为关系增加核验请求，关系多时需要多个核验批次，失败恢复还会
增加请求。`openie_strict=false` 只控制失败后的发布行为；降低核验开销应选择
`structural`。实际耗时还取决于原文长度、重试数量、缓存命中和服务吞吐，不能仅按模式
名称保证固定完成时间。
