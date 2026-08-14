# V032-R1 gain diagnostics

`events.jsonl` is produced by `experiments/run_v032_gain_validation.py` from
local executable notebook opponents only. Each row compares V27 order-only
control with one forced old-V032 timing event. The fixed `nowinlog` replay set
is external validation and is not used to fit this calibration.

`summary.json` reports raw prediction MAE, residual-corrected MAE,
positive-prediction/negative-outcome events, and leave-one-source-out
diagnostics.

Calibration is embedded into a submission only when its support-group gate is
met. Small smoke samples remain diagnostics and intentionally produce a
control-only timing candidate.
