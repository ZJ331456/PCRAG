"""Offline OpenIE backend adapter."""

from .offline import OfflineOpenIE


class TransformersOfflineOpenIE(OfflineOpenIE):
    def __init__(self, global_config):
        from ...llm.transformers_offline import TransformersOffline

        self._configure_prompts(getattr(global_config, 'openie_prompt_version', 'optimized'))
        self.llm_model = TransformersOffline(global_config)
