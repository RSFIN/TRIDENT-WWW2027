```python
 uv run python .\scripts\run_cicc.py --config .\configs\cicc_clinc150.json
========================================================================================
CICC experiment results | smoke=false | seeds=[42, 43, 44]
========================================================================================
Samples: hard_negative_test=0, oos_test=1000, oos_val=100, test=4500, train=15000, val=3000

Seed 42
----------------------------------------------------------------------------------------
[L1 Closed-set]
  Accuracy                                          0.629333
  Macro-F1                                          0.625303
[L2 OOS]
  AUROC                                             0.758817
  AUPRC                                             0.330718
  FPR95                                             0.624222
  F1-OOS (CICC decision)                            0.378211
  F1-OOS (tuned threshold)                          0.409170
[L3 Calibration]
  ECE (15 bins)                                     0.163491
  Multiclass Brier                                  0.562013
  Empirical coverage                                0.989556
  Nominal coverage                                  0.800000
  Coverage gap                                      0.189556
  Average prediction-set size                      77.558889
[L4 Efficiency]
  Latency P50 (ms)                                  0.129500
  Latency P99 (ms)                                  0.275417
  Index build time (s)                              0.073885
  Tokens/sample (approx.)                           8.313818
  Trainable parameters                                     0
[L5 Decision]
  CICC reject rate                                  0.759455
  CICC false-reject rate                            0.710667
  Reject rate @ target OOS recall                   0.650909
  False-reject rate @ target OOS recall             0.587333
  Realized OOS recall                               0.937000
  Clarification rate (in-scope)                     0.090222
  Clarification-set coverage                        0.953202
  Mean clarification rounds                         0.090222
  Hard-negative OOS                         not evaluated
  Clarification-trigger accuracy                         N/A
  Turn-valid multi-round evaluated                        no

Seed 43
----------------------------------------------------------------------------------------
[L1 Closed-set]
  Accuracy                                          0.629333
  Macro-F1                                          0.625303
[L2 OOS]
  AUROC                                             0.759284
  AUPRC                                             0.332360
  FPR95                                             0.624667
  F1-OOS (CICC decision)                            0.377525
  F1-OOS (tuned threshold)                          0.409260
[L3 Calibration]
  ECE (15 bins)                                     0.163491
  Multiclass Brier                                  0.562013
  Empirical coverage                                0.989778
  Nominal coverage                                  0.800000
  Coverage gap                                      0.189778
  Average prediction-set size                      78.401778
[L4 Efficiency]
  Latency P50 (ms)                                  0.129700
  Latency P99 (ms)                                  0.299918
  Index build time (s)                              0.073885
  Tokens/sample (approx.)                           8.313818
  Trainable parameters                                     0
[L5 Decision]
  CICC reject rate                                  0.763091
  CICC false-reject rate                            0.714667
  Reject rate @ target OOS recall                   0.650727
  False-reject rate @ target OOS recall             0.587111
  Realized OOS recall                               0.937000
  Clarification rate (in-scope)                     0.090444
  Clarification-set coverage                        0.950860
  Mean clarification rounds                         0.090444
  Hard-negative OOS                         not evaluated
  Clarification-trigger accuracy                         N/A
  Turn-valid multi-round evaluated                        no

Seed 44
----------------------------------------------------------------------------------------
[L1 Closed-set]
  Accuracy                                          0.629333
  Macro-F1                                          0.625303
[L2 OOS]
  AUROC                                             0.758979
  AUPRC                                             0.331912
  FPR95                                             0.624889
  F1-OOS (CICC decision)                            0.374785
  F1-OOS (tuned threshold)                          0.408368
[L3 Calibration]
  ECE (15 bins)                                     0.163491
  Multiclass Brier                                  0.562013
  Empirical coverage                                0.990222
  Nominal coverage                                  0.800000
  Coverage gap                                      0.190222
  Average prediction-set size                      80.133111
[L4 Efficiency]
  Latency P50 (ms)                                  0.128200
  Latency P99 (ms)                                  0.270908
  Index build time (s)                              0.073885
  Tokens/sample (approx.)                           8.313818
  Trainable parameters                                     0
[L5 Decision]
  CICC reject rate                                  0.770000
  CICC false-reject rate                            0.723111
  Reject rate @ target OOS recall                   0.652545
  False-reject rate @ target OOS recall             0.589333
  Realized OOS recall                               0.937000
  Clarification rate (in-scope)                     0.089556
  Clarification-set coverage                        0.955335
  Mean clarification rounds                         0.089556
  Hard-negative OOS                         not evaluated
  Clarification-trigger accuracy                         N/A
  Turn-valid multi-round evaluated                        no

Aggregate across seeds
----------------------------------------------------------------------------------------
Metric                                                 Mean        Std              95% CI
----------------------------------------------------------------------------------------
Accuracy                                           0.629333   0.000000 [0.629333, 0.629333]
Macro-F1                                           0.625303   0.000000 [0.625303, 0.625303]
AUROC                                              0.759027   0.000237 [0.758437, 0.759616]
AUPRC                                              0.331664   0.000849 [0.329555, 0.333772]
FPR95                                              0.624593   0.000339 [0.623749, 0.625436]
F1-OOS (CICC decision)                             0.376841   0.001813 [0.372337, 0.381344]
F1-OOS (tuned threshold)                           0.408933   0.000491 [0.407713, 0.410153]
ECE (15 bins)                                      0.163491   0.000000 [0.163491, 0.163491]
Multiclass Brier                                   0.562013   0.000000 [0.562013, 0.562013]
Empirical coverage                                 0.989852   0.000339 [0.989009, 0.990695]
Nominal coverage                                   0.800000   0.000000 [0.800000, 0.800000]
Coverage gap                                       0.189852   0.000339 [0.189009, 0.190695]
Average prediction-set size                       78.697926   1.312415 [75.437707, 81.958145]
Latency P50 (ms)                                   0.129133   0.000814 [0.127110, 0.131157]
Latency P99 (ms)                                   0.282081   0.015611 [0.243301, 0.320861]
Index build time (s)                               0.073885   0.000000 [0.073885, 0.073885]
Tokens/sample (approx.)                            8.313818   0.000000 [8.313818, 8.313818]
Trainable parameters                               0.000000   0.000000 [0.000000, 0.000000]
CICC reject rate                                   0.764182   0.005357 [0.750875, 0.777489]
CICC false-reject rate                             0.716148   0.006353 [0.700366, 0.731930]
Reject rate @ target OOS recall                    0.651394   0.001001 [0.648906, 0.653881]
False-reject rate @ target OOS recall              0.587926   0.001224 [0.584886, 0.590966]
Realized OOS recall                                0.937000   0.000000 [0.937000, 0.937000]
Clarification rate (in-scope)                      0.090074   0.000463 [0.088925, 0.091223]
Clarification-set coverage                         0.953132   0.002238 [0.947572, 0.958693]
Mean clarification rounds                          0.090074   0.000463 [0.088925, 0.091223]
========================================================================================
Artifacts: D:\Project\PythonProject\www2027-intent\cicc-reproduction\results\cicc_clinc150
```

