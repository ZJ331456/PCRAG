# 索引构建代码

## 目录

| 位置 | 职责 |
| --- | --- |
| `ner/` | 在线与离线命名实体抽取、NER 校验和重试 |
| `openie/` | 三元组抽取、关系校验、原文证据核验和恢复 |
| `shared_index_builder.py` | 共享索引构建入口及 HippoRAG 运行时适配 |
| `openie_build_queue.py` | 协调 NER 和三元组阶段、控制并发、恢复未完成片段 |
| `openie_checkpoint.py` | 保存和读取逐阶段 SQLite 检查点 |
| `extraction_utils.py` | 两阶段共用的响应解析、环境变量和重试设置 |

在线提取由 `OpenIE` 组合 NER 与三元组阶段；严格构建使用
`SourceVerifiedOpenIE`，NER 输出上限保持 512，三元组输出上限保持 2048。
离线后端共用批量 NER 与 OpenIE 实现。

`scripts/build_shared_index.py` 仍是命令行入口。
信息抽取实现统一位于 `ner/` 与 `openie/`，项目内部均引用这些新路径。
离线模型依赖只在构造相应后端时加载。

## 失败策略与提示词

构建入口可以追加以下参数，`eval_dataset.py` 和新索引对比脚本也支持它们：

```bash
--openie_strict true --openie_prompt_version optimized
```

- `openie_strict=true`（默认）：所有 chunk 都必须完成 NER 和最终关系核验，
  未完成则保存诊断并中止图发布。经独立原文核验确认没有关系的文本允许为空；
  此开关不等于要求任何文本都生成非空关系，也不保证 LLM 没有语义错误。
- `openie_strict=false`：重试与核验流程保持一致，最终失败会警告并记录统计，
  对应 chunk 的文本和向量保留，未通过检查的关系不入图，继续完成索引。
  该选项只放宽抽取失败；语料身份冲突、损坏检查点等错误仍会中止。
- `openie_prompt_version=origin`：使用此次优化前保存的
  `prompts/templates/origin_ner_prompt.py` 和 `origin_triple_extraction_prompt.py`。
- `openie_prompt_version=optimized`（默认）：使用优化后的 `ner.py` 和
  `triple_extraction.py`。`origin` 指本项目优化前版本，不是承诺与原生 HippoRAG
  提示词完全相同。

失败策略、所选提示词和提示词哈希写入质量身份。变更后应在独立目录重建索引，
不把旧检查点误当作新提示词的实验结果。原始 HippoRAG 源码不作修改。

## 开销

NER 和三元组各阶段最多 8 个 worker，保持 NER 512、三元组/核验 2048，
Qwen3 thinking 关闭。严格抽取从第一次三元组请求就约束 JSON 结构，避免先生成
错误任务键或四/五字段结果再纠正。原文证据核验保持开启；JSON 合法不代表事实正确。
预检查和实际请求重复使用相同提示词时缓存精确 token 计数，语法编译也有缓存。
成功检查点可以跳过已经完成的阶段，最终失败的 chunk 才进入额外恢复。
失败 NER 原文片段会继续按原始边界细分，成功片段可恢复复用；最多 64 次累计
NER 请求、12 层细分，只有覆盖全部原文时才标记完整。恢复反馈带尝试编号，避免
相同失败反馈一直命中同一个失败响应缓存。固定长度数组通过等价 `prefixItems`
编译，绕开当前 xgrammar 对 `minItems/maxItems` 的忽略。

每个 chunk 的 NER、三元组抽取加关系核验通常比原生 HippoRAG 多 LLM 请求，
关系较多还需要多个核验批次。上述改动减少格式错误和重复计算，不能消除核验本身
的模型耗时。`openie_strict=false` 也不等于关闭核验，因此不会自动得到原生速度。
