from .base import EmbeddingConfig, BaseEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        from .Contriever import ContrieverModel
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        from .OpenAI import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel
    elif "qwen-embedding" in embedding_model_name.lower():
        # OpenAI-compatible embedding endpoint with qwen served model name.
        from .OpenAI import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        from .Cohere import CohereEmbeddingModel
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/") or "Qwen3-Embedding" in embedding_model_name:
        from .Transformers import TransformersEmbeddingModel
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        from .VLLM import VLLMEmbeddingModel
        return VLLMEmbeddingModel
    elif (
        embedding_model_name.startswith("TEI/")
        or embedding_model_name.startswith("tei/")
        or embedding_model_name.startswith("TEI:")
        or embedding_model_name.startswith("tei:")
    ):
        # Hugging Face Text Embeddings Inference HTTP backend.
        from .TEI import TEIEmbeddingModel
        return TEIEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"