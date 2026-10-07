"""Compose independently switchable extensions to the online exp4 algorithm."""

from .evidence_binding import BindingImprovementMixin
from .evidence_planning import AdaptiveSearchMixin, PlannerImprovementMixin
from .evidence_retrieval import EvidenceRetrieval
from .evidence_selection import SelectionImprovementMixin
from .evidence_support import AncestorSupportMixin
from .evidence_terminal import TerminalEvidenceMixin


class ImprovedEvidenceRetrieval(
    PlannerImprovementMixin,
    AdaptiveSearchMixin,
    BindingImprovementMixin,
    TerminalEvidenceMixin,
    AncestorSupportMixin,
    SelectionImprovementMixin,
    EvidenceRetrieval,
):
    def __init__(self, rag):
        self.improvements = frozenset(
            f.strip() for f in rag.pcrag_config.evidence_improvements.split(",") if f.strip()
        )
        super().__init__(rag)
