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
    - subspace_patching: Subspace activation patching (Eq. 1 of Makelov et al.,
      2023) plus the ker/rowspace diagnostics that separate a faithful
      subspace from an illusory one.
"""

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
from transformer_lens.tools.analysis.subspace_patching import (
    SubspacePatchReport,
    fractional_logit_diff_decrease,
    nullspace_rowspace_decomposition,
    projection_spread,
    subspace_patch_faithfulness,
    subspace_patch_setter,
)

__all__ = [
    "DirectLogitAttribution",
    "JacobianLens",
    "JacobianLensReadout",
    "SubspacePatchReport",
    "direct_logit_attribution",
    "fractional_logit_diff_decrease",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "nullspace_rowspace_decomposition",
    "projection_spread",
    "subspace_patch_faithfulness",
    "subspace_patch_setter",
]
