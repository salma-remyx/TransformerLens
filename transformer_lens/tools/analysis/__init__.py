"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. Model support is documented per tool;
new analyses may target the ``TransformerBridge`` API exclusively.

Tools:
    - direct_logit_attribution: Direct Logit Attribution (DLA) over components,
      layers, or attention heads.
    - direct_path_patching: Direct path patching for head-to-head circuit
      analysis.
    - jacobian_lens: The Jacobian lens (J-lens) — per-layer causal transport to
      the output vocabulary basis, with loading of published lens artifacts,
      native fitting, readouts, and interventions.
    - activation_transfer: Cross-model transfer of activation states —
      representational similarity, learned projection with cross-model
      retrieval, and causal residual injection into a second model.
"""

from transformer_lens.tools.analysis.activation_transfer import (
    ActivationProjection,
    ActivationTransferReport,
    collect_paired_activations,
    cross_model_transfer_report,
    linear_cka,
    mutual_knn_alignment,
    residual_injection_hooks,
)
from transformer_lens.tools.analysis.direct_logit_attribution import (
    DirectLogitAttribution,
    direct_logit_attribution,
)
from transformer_lens.tools.analysis.direct_path_patching import (
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)
from transformer_lens.tools.analysis.jacobian_lens import (
    JacobianLens,
    JacobianLensReadout,
)

__all__ = [
    "ActivationProjection",
    "ActivationTransferReport",
    "DirectLogitAttribution",
    "JacobianLens",
    "JacobianLensReadout",
    "collect_paired_activations",
    "cross_model_transfer_report",
    "direct_logit_attribution",
    "linear_cka",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "mutual_knn_alignment",
    "residual_injection_hooks",
]
