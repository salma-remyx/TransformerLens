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
    - visual_grounding: Visual lookback scoring — how much attention mass each
      response token places on the image span, combined with token likelihood
      for Best-of-N selection in vision-language models.
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
from transformer_lens.tools.analysis.visual_grounding import (
    LookBackScore,
    VisualLookback,
    find_image_span,
    lookback_score,
    select_best_of_n,
    visual_lookback,
)

__all__ = [
    "DirectLogitAttribution",
    "JacobianLens",
    "JacobianLensReadout",
    "LookBackScore",
    "VisualLookback",
    "direct_logit_attribution",
    "find_image_span",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "lookback_score",
    "select_best_of_n",
    "visual_lookback",
]
