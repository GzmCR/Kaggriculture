# Stable12: pure fixed route

This folder archives a standalone route-only submission extracted from the
local Kaggriculture research repository.

Runtime behavior:

- return the frozen action trace for the current `step`;
- keep market orders exactly as embedded in the trace;
- do not reorder or add market orders;
- do not use WEED recovery, opponent state, replay files, notebooks, scores,
  seeds, network access, or external APIs;
- do not add terminal liquidation or dynamic hand alignment.

Route length: `720` actions  
Route SHA-256: `f23f023fdfe06fe45724a103569f886c7e2642d1bce8cbc5e3818b04eb009ced`

The route was extracted from the source recorded in the repository manifest.
It is archived for pure production-route comparison, not selected as the
current root `main.py`.
