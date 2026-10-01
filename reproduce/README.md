# HippoRAG2 本地检索适配补丁 / Local retrieval patches

HippoRAG 源码基准提交：`1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff`。

- `hipporag_local_runtime.patch`：相对该提交的完整本地源码修改，包含原有 Qwen3 embedding 适配、本次详报、并发、请求统计、错误传播、从空目录完整建索引及 OpenIE 截断校验。用于干净的该版本 checkout。
- `hipporag_detailed_retrieval.patch`：仅本次相对修改前本地源码的差异；用于已有本地 Qwen3 适配的副本。两个补丁选一个应用。

```bash
cd /path/to/HippoRAG
git apply --check /path/to/PathCondRAG/reproduce/hipporag_local_runtime.patch
git apply /path/to/PathCondRAG/reproduce/hipporag_local_runtime.patch
```

本机 `/root/baseline/HippoRAG` 已应用源码修改，无需再次应用。上游 HippoRAG 仓库不会推送；补丁随 PCRAG 提交保存。实验细节见 `docs/recall_improvement_experiments.md`。

`hipporag_patch_manifest.json` 记录基准提交及增量补丁涉及文件的修改前后 SHA256。两个补丁都已在临时目录通过 `git apply --check`，应用后字节内容与当前源码一致。

## 从空目录构建同一份共享索引

```bash
export HIPPORAG_LLM_MAX_IN_FLIGHT=8
export PATHCONDRAG_LLM_MAX_IN_FLIGHT=8
python /root/baseline/HippoRAG/main.py \
  --dataset musique --datasets_dir /root/datasets \
  --embedding_name /root/models/Qwen3-Embedding-8B \
  --embedding_provider transformers --embedding_batch_size 4 \
  --llm_name qwen3-8b --llm_base_url http://127.0.0.1:8035/v1 \
  --llm_prefetch_workers 8 --openie_max_workers 8 \
  --force_index_from_scratch true --force_openie_from_scratch true \
  --save_dir /path/to/new-empty/shared_hipporag2_index --save_dir_exact \
  --eval_mode index_only --output /path/to/output/index_build_result.json
```

`index_only` 要求新建且为空的 `save_dir`，不复用任何旧图、向量、OpenIE 或 SQLite 响应缓存，也不执行检索和 QA。NER 与三元组阶段各使用 8 个 worker，共享进程级最多 8 个 HTTP 请求。全局输出上限和三元组输出上限保持 2048；NER 保留 512→1024 的原有预算。

`finish_reason=length` 的响应不会修补后标为成功。只在发生截断时，按同一最终预算进行有限的 seed 与频率惩罚重试；仍然截断则阻止建索引。每篇 OpenIE 保存 `openie_metadata.ner` 和 `openie_metadata.triples`，包括最终 finish reason、缓存信息和重试设置。合法空实体或三元组列表会单独统计；HTTP 失败、非正常终止、无效抽取及缺失文档不会生成成功报告。

建索引成功报告保存实际 corpus/chunk/OpenIE 数量、耗时、完整配置、真实 LLM 请求统计与 `index_build_complete=true`。后续七组实验均以这份新索引为源，使用 `--reuse_index` 或等价的只读检索路径。
