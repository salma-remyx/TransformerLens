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
    - world_model_probe: Linear probes for a task's state space — fit a probe,
      measure how the representation decays along a sequence, and restore the
      prompt-time representation at inference.
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
from transformer_lens.tools.analysis.world_model_probe import (
    ProbeFidelity,
    WorldModelProbe,
    fit_world_model_probe,
    probe_fidelity_by_position,
    restoration_hooks,
)

__all__ = [
    "DirectLogitAttribution",
    "JacobianLens",
    "JacobianLensReadout",
    "ProbeFidelity",
    "WorldModelProbe",
    "direct_logit_attribution",
    "fit_world_model_probe",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "probe_fidelity_by_position",
    "restoration_hooks",
]
