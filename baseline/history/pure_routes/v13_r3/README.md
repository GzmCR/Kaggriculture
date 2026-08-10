# V13_R3: pure fixed route

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
Route SHA-256: `9d46e73be2ab901ca6fd0384221b8f42e2407cefe40cde01ab3130371b217462`

The route was extracted from the source recorded in the repository manifest.
It is archived for pure production-route comparison, not selected as the
current root `main.py`.
