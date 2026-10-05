"""Offline OpenIE backend adapter."""

from .offline import OfflineOpenIE


class VLLMOfflineOpenIE(OfflineOpenIE):
    def __init__(self, global_config):
        from ...llm.vllm_offline import VLLMOffline

        self._configure_prompts(getattr(global_config, 'openie_prompt_version', 'optimized'))
        self.llm_model = VLLMOffline(global_config)
