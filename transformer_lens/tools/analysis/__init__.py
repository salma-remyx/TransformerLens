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
    - subspace_patching: Subspace-constrained activation patching, plus the
      bidirectional asymmetry diagnostic that separates subspaces which carry
      a feature from those whose apparent effect is illusory.
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
    SubspacePatchSweeps,
    get_act_patch_resid_subspace_all_pos,
    layer_pos_subspace_patch_setter,
    subspace_patch_asymmetry,
)

__all__ = [
    "DirectLogitAttribution",
    "JacobianLens",
    "JacobianLensReadout",
    "SubspacePatchSweeps",
    "direct_logit_attribution",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "get_act_patch_resid_subspace_all_pos",
    "layer_pos_subspace_patch_setter",
    "subspace_patch_asymmetry",
]
