# HippoRAG2 本地检索适配补丁 / Local retrieval patches

HippoRAG 源码基准提交：`1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff`。

- `hipporag_local_runtime.patch`：相对该提交的完整本地源码修改，包含原有 Qwen3 embedding 适配及本次详报、并发、请求统计和错误传播。用于干净的该版本 checkout。
- `hipporag_detailed_retrieval.patch`：仅本次相对修改前本地源码的差异；用于已有本地 Qwen3 适配的副本。两个补丁选一个应用。

```bash
cd /path/to/HippoRAG
git apply --check /path/to/PathCondRAG/reproduce/hipporag_local_runtime.patch
git apply /path/to/PathCondRAG/reproduce/hipporag_local_runtime.patch
```

本机 `/root/baseline/HippoRAG` 已应用源码修改，无需再次应用。上游 HippoRAG 仓库不会推送；补丁随 PCRAG 提交保存。实验细节见 `docs/recall_improvement_experiments.md`。
