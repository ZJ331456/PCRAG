"""Compose independently switchable extensions to the online exp4 algorithm."""

from .evidence_binding import BindingImprovementMixin
from .evidence_anchor_companion import AnchorCompanionMixin
from .evidence_condition_companion import ConditionCompanionMixin
from .evidence_bridge import BridgeRecoveryMixin
from .evidence_dag_package import DAGPackageMixin
from .evidence_package import EvidencePackageMixin
from .evidence_planning import AdaptiveSearchMixin, PlannerImprovementMixin
from .evidence_retrieval import EvidenceRetrieval
from .evidence_selection import SelectionImprovementMixin
from .evidence_support import AncestorSupportMixin
from .evidence_terminal import TerminalEvidenceMixin
from .evidence_structural_recovery import StructuralRecoveryMixin
from .evidence_source_witness import SourceWitnessMixin
from .evidence_failure_recovery import FailureRecoveryMixin


class ImprovedEvidenceRetrieval(
    FailureRecoveryMixin,
    SourceWitnessMixin,
    StructuralRecoveryMixin,
    BridgeRecoveryMixin,
    PlannerImprovementMixin,
    AdaptiveSearchMixin,
    BindingImprovementMixin,
    ConditionCompanionMixin,
    AnchorCompanionMixin,
    DAGPackageMixin,
    EvidencePackageMixin,
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
