# V22: pure fixed route

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
Route SHA-256: `cd4380d55c4a13c2ed4fd0c9463268c5764599f7e3f58b91e960b49d7dfd5d77`

The route was extracted from the source recorded in the repository manifest.
It is archived for pure production-route comparison, not selected as the
current root `main.py`.
