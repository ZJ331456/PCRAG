"""Offline OpenIE backend adapter."""

from .offline import OfflineOpenIE
from ...prompts import PromptTemplateManager


class TransformersOfflineOpenIE(OfflineOpenIE):
    def __init__(self, global_config):
        from ...llm.transformers_offline import TransformersOffline

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = TransformersOffline(global_config)
