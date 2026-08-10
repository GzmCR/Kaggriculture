# Adaptive_V14: pure fixed route

This folder archives a standalone route-only submission extracted from the
local Kaggriculture research repository.

Runtime behavior:

- return the frozen action trace for the current `step`;
- keep market orders exactly as embedded in the trace;
- do not reorder or add market orders;
- do not use WEED recovery, opponent state, replay files, notebooks, scores,
  seeds, network access, or external APIs;
- do not add terminal liquidation or dynamic hand alignment.

Route length: `719` actions  
Route SHA-256: `04354aa03513bfc1dcfbbeaf25a4a2074317681bd3c839a9550db21fa3e0d406`

The route was extracted from the source recorded in the repository manifest.
It is archived for pure production-route comparison, not selected as the
current root `main.py`.
