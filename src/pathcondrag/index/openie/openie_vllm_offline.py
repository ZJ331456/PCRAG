"""Offline OpenIE backend adapter."""

from .offline import OfflineOpenIE
from ...prompts import PromptTemplateManager


class VLLMOfflineOpenIE(OfflineOpenIE):
    def __init__(self, global_config):
        from ...llm.vllm_offline import VLLMOffline

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)
