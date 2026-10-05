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
共享索引的历史 producer 标签和质量合同保持不变，保证原有检查点可继续核验。
离线模型依赖只在构造相应后端时加载。
