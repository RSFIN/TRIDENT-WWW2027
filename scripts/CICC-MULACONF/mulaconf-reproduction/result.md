##  CLINC150

```python
(mulaconf-reproduction) PS D:\Project\PythonProject\www2027-intent\mulaconf-reproduction> uv run python .\scripts\run_mulaconf.py `
>>   --config .\configs\mulaconf_all.json `
>>   --dataset clinc150
====================================================================================================
Multi-label conformal results | smoke=false | seeds=[42, 43, 44]
====================================================================================================

Dataset: clinc150 | singleton labelsets
----------------------------------------------------------------------------------------------------
Samples: atomic_labels=150, observed_labelsets=150, oos_test=1000, oos_val=100, test=4500, train=15000, val=3000

====================================================================================================
Method: powerset
====================================================================================================

Seed 42
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.629333
  Macro-F1                                            0.625303
  Intent-set IoU                                      0.629333
  Hit@1                                               0.629333
  Hit@3                                               0.833111
  Hit@5                                               0.894667
[L2 OOS]
  AUROC                                               0.768102
  AUPRC                                               0.367958
  FPR95                                               0.637111
  F1-OOS (region decision)                            0.396546
  F1-OOS (tuned threshold)                            0.394185
[L3 Calibration]
  ECE (15 bins)                                       0.163491
  Brier (labelset)                                    0.562013
  Exact labelset coverage                             0.915333
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.915333
  Jaccard>=0.5 coverage                               0.915333
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.015333
  Average prediction-region size                      4.531333
[L4 Efficiency]
  Latency P50 (ms)                                    0.110000
  Latency P99 (ms)                                    0.295322
  Tokens/sample (approx.)                             8.313818
  Trainable parameters                                       0
  Index build time (s)                                0.072443
[L5 Decision]
  Reject rate                                         0.365636
  False-reject rate                                   0.314222
  Clarification rate                                  0.467333
  Clarification Jaccard coverage                      0.906324
  Mean clarification rounds                           0.467333
  Set-level Jaccard error rate                        0.370667
  Reject rate @ 95% OOS recall                        0.693636
  False reject @ 95% OOS recall                       0.636889
  Realized OOS recall                                 0.949000
  Tuned OOS threshold                                 0.343103
  Hard-negative OOS                              not evaluated
  Clarification-trigger accuracy                         N/A

Seed 43
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.629333
  Macro-F1                                            0.625303
  Intent-set IoU                                      0.629333
  Hit@1                                               0.629333
  Hit@3                                               0.833111
  Hit@5                                               0.894667
[L2 OOS]
  AUROC                                               0.768102
  AUPRC                                               0.367958
  FPR95                                               0.637111
  F1-OOS (region decision)                            0.404414
  F1-OOS (tuned threshold)                            0.394185
[L3 Calibration]
  ECE (15 bins)                                       0.163491
  Brier (labelset)                                    0.562013
  Exact labelset coverage                             0.917556
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.917556
  Jaccard>=0.5 coverage                               0.917556
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.017556
  Average prediction-region size                      4.661778
[L4 Efficiency]
  Latency P50 (ms)                                    0.107800
  Latency P99 (ms)                                    0.224700
  Tokens/sample (approx.)                             8.313818
  Trainable parameters                                       0
  Index build time (s)                                0.072443
[L5 Decision]
  Reject rate                                         0.378364
  False-reject rate                                   0.324000
  Clarification rate                                  0.461333
  Clarification Jaccard coverage                      0.911368
  Mean clarification rounds                           0.461333
  Set-level Jaccard error rate                        0.370667
  Reject rate @ 95% OOS recall                        0.693636
  False reject @ 95% OOS recall                       0.636889
  Realized OOS recall                                 0.949000
  Tuned OOS threshold                                 0.343103
  Hard-negative OOS                              not evaluated
  Clarification-trigger accuracy                         N/A

Seed 44
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.629333
  Macro-F1                                            0.625303
  Intent-set IoU                                      0.629333
  Hit@1                                               0.629333
  Hit@3                                               0.833111
  Hit@5                                               0.894667
[L2 OOS]
  AUROC                                               0.768102
  AUPRC                                               0.367958
  FPR95                                               0.637111
  F1-OOS (region decision)                            0.387230
  F1-OOS (tuned threshold)                            0.394185
[L3 Calibration]
  ECE (15 bins)                                       0.163491
  Brier (labelset)                                    0.562013
  Exact labelset coverage                             0.911333
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.911333
  Jaccard>=0.5 coverage                               0.911333
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.011333
  Average prediction-region size                      4.329333
[L4 Efficiency]
  Latency P50 (ms)                                    0.107300
  Latency P99 (ms)                                    0.246692
  Tokens/sample (approx.)                             8.313818
  Trainable parameters                                       0
  Index build time (s)                                0.072443
[L5 Decision]
  Reject rate                                         0.347818
  False-reject rate                                   0.299778
  Clarification rate                                  0.477333
  Clarification Jaccard coverage                      0.902700
  Mean clarification rounds                           0.477333
  Set-level Jaccard error rate                        0.370667
  Reject rate @ 95% OOS recall                        0.693636
  False reject @ 95% OOS recall                       0.636889
  Realized OOS recall                                 0.949000
  Tuned OOS threshold                                 0.343103
  Hard-negative OOS                              not evaluated
  Clarification-trigger accuracy                         N/A

Final aggregate across 3 seed(s)
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
Metric                                             Mean         Std                    95% CI
Exact Match                                    0.629333    0.000000      [0.629333, 0.629333]
Macro-F1                                       0.625303    0.000000      [0.625303, 0.625303]
Intent-set IoU                                 0.629333    0.000000      [0.629333, 0.629333]
Hit@1                                          0.629333    0.000000      [0.629333, 0.629333]
Hit@3                                          0.833111    0.000000      [0.833111, 0.833111]
Hit@5                                          0.894667    0.000000      [0.894667, 0.894667]
[L2 OOS]
Metric                                             Mean         Std                    95% CI
AUROC                                          0.768102    0.000000      [0.768102, 0.768102]
AUPRC                                          0.367958    0.000000      [0.367958, 0.367958]
FPR95                                          0.637111    0.000000      [0.637111, 0.637111]
F1-OOS (region decision)                       0.396063    0.008602      [0.374694, 0.417433]
F1-OOS (tuned threshold)                       0.394185    0.000000      [0.394185, 0.394185]
[L3 Calibration]
Metric                                             Mean         Std                    95% CI
ECE (15 bins)                                  0.163491    0.000000      [0.163491, 0.163491]
Brier (labelset)                               0.562013    0.000000      [0.562013, 0.562013]
Exact labelset coverage                        0.914741    0.003153      [0.906908, 0.922574]
Seen-labelset rate                             1.000000    0.000000      [1.000000, 1.000000]
Coverage on seen labelsets                     0.914741    0.003153      [0.906908, 0.922574]
Jaccard>=0.5 coverage                          0.914741    0.003153      [0.906908, 0.922574]
Nominal coverage                               0.900000    0.000000      [0.900000, 0.900000]
Exact coverage gap                             0.014741    0.003153      [0.006908, 0.022574]
Average prediction-region size                 4.507481    0.167501      [4.091386, 4.923576]
[L4 Efficiency]
Metric                                             Mean         Std                    95% CI
Latency P50 (ms)                               0.108367    0.001436      [0.104798, 0.111935]
Latency P99 (ms)                               0.255571    0.036139      [0.165798, 0.345345]
Tokens/sample (approx.)                        8.313818    0.000000      [8.313818, 8.313818]
Trainable parameters                           0.000000    0.000000      [0.000000, 0.000000]
Index build time (s)                           0.072443    0.000000      [0.072443, 0.072443]
[L5 Decision]
Metric                                             Mean         Std                    95% CI
Reject rate                                    0.363939    0.015343      [0.325825, 0.402054]
False-reject rate                              0.312667    0.012186      [0.282395, 0.342938]
Clarification rate                             0.468667    0.008083      [0.448588, 0.488746]
Clarification Jaccard coverage                 0.906798    0.004353      [0.895983, 0.917612]
Mean clarification rounds                      0.468667    0.008083      [0.448588, 0.488746]
Set-level Jaccard error rate                   0.370667    0.000000      [0.370667, 0.370667]
Reject rate @ 95% OOS recall                   0.693636    0.000000      [0.693636, 0.693636]
False reject @ 95% OOS recall                  0.636889    0.000000      [0.636889, 0.636889]
Realized OOS recall                            0.949000    0.000000      [0.949000, 0.949000]
Tuned OOS threshold                            0.343103    0.000000      [0.343103, 0.343103]

====================================================================================================
Method: tree_fixed_alpha
====================================================================================================

Seed 42
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.629333
  Macro-F1                                            0.625303
  Intent-set IoU                                      0.629333
  Hit@1                                               0.629333
  Hit@3                                               0.833111
  Hit@5                                               0.894667
[L2 OOS]
  AUROC                                               0.768102
  AUPRC                                               0.367958
  FPR95                                               0.637111
  F1-OOS (region decision)                            0.307692
  F1-OOS (tuned threshold)                            0.394185
[L3 Calibration]
  ECE (15 bins)                                       0.163491
  Brier (labelset)                                    0.562013
  Exact labelset coverage                             0.996222
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.996222
  Jaccard>=0.5 coverage                               0.996222
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.096222
  Average prediction-region size                    133.216889
[L4 Efficiency]
  Latency P50 (ms)                                    0.110000
  Latency P99 (ms)                                    0.295322
  Tokens/sample (approx.)                             8.313818
  Trainable parameters                                       0
  Index build time (s)                                0.072443
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.370667
  Reject rate @ 95% OOS recall                        0.693636
  False reject @ 95% OOS recall                       0.636889
  Realized OOS recall                                 0.949000
  Tuned OOS threshold                                 0.343103
  Hard-negative OOS                              not evaluated
  Clarification-trigger accuracy                         N/A

Seed 43
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.629333
  Macro-F1                                            0.625303
  Intent-set IoU                                      0.629333
  Hit@1                                               0.629333
  Hit@3                                               0.833111
  Hit@5                                               0.894667
[L2 OOS]
  AUROC                                               0.768102
  AUPRC                                               0.367958
  FPR95                                               0.637111
  F1-OOS (region decision)                            0.307692
  F1-OOS (tuned threshold)                            0.394185
[L3 Calibration]
  ECE (15 bins)                                       0.163491
  Brier (labelset)                                    0.562013
  Exact labelset coverage                             0.996444
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.996444
  Jaccard>=0.5 coverage                               0.996444
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.096444
  Average prediction-region size                    134.715778
[L4 Efficiency]
  Latency P50 (ms)                                    0.107800
  Latency P99 (ms)                                    0.224700
  Tokens/sample (approx.)                             8.313818
  Trainable parameters                                       0
  Index build time (s)                                0.072443
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.370667
  Reject rate @ 95% OOS recall                        0.693636
  False reject @ 95% OOS recall                       0.636889
  Realized OOS recall                                 0.949000
  Tuned OOS threshold                                 0.343103
  Hard-negative OOS                              not evaluated
  Clarification-trigger accuracy                         N/A

Seed 44
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.629333
  Macro-F1                                            0.625303
  Intent-set IoU                                      0.629333
  Hit@1                                               0.629333
  Hit@3                                               0.833111
  Hit@5                                               0.894667
[L2 OOS]
  AUROC                                               0.768102
  AUPRC                                               0.367958
  FPR95                                               0.637111
  F1-OOS (region decision)                            0.307692
  F1-OOS (tuned threshold)                            0.394185
[L3 Calibration]
  ECE (15 bins)                                       0.163491
  Brier (labelset)                                    0.562013
  Exact labelset coverage                             0.997778
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.997778
  Jaccard>=0.5 coverage                               0.997778
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.097778
  Average prediction-region size                    137.328667
[L4 Efficiency]
  Latency P50 (ms)                                    0.107300
  Latency P99 (ms)                                    0.246692
  Tokens/sample (approx.)                             8.313818
  Trainable parameters                                       0
  Index build time (s)                                0.072443
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.370667
  Reject rate @ 95% OOS recall                        0.693636
  False reject @ 95% OOS recall                       0.636889
  Realized OOS recall                                 0.949000
  Tuned OOS threshold                                 0.343103
  Hard-negative OOS                              not evaluated
  Clarification-trigger accuracy                         N/A

Final aggregate across 3 seed(s)
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
Metric                                             Mean         Std                    95% CI
Exact Match                                    0.629333    0.000000      [0.629333, 0.629333]
Macro-F1                                       0.625303    0.000000      [0.625303, 0.625303]
Intent-set IoU                                 0.629333    0.000000      [0.629333, 0.629333]
Hit@1                                          0.629333    0.000000      [0.629333, 0.629333]
Hit@3                                          0.833111    0.000000      [0.833111, 0.833111]
Hit@5                                          0.894667    0.000000      [0.894667, 0.894667]
[L2 OOS]
Metric                                             Mean         Std                    95% CI
AUROC                                          0.768102    0.000000      [0.768102, 0.768102]
AUPRC                                          0.367958    0.000000      [0.367958, 0.367958]
FPR95                                          0.637111    0.000000      [0.637111, 0.637111]
F1-OOS (region decision)                       0.307692    0.000000      [0.307692, 0.307692]
F1-OOS (tuned threshold)                       0.394185    0.000000      [0.394185, 0.394185]
[L3 Calibration]
Metric                                             Mean         Std                    95% CI
ECE (15 bins)                                  0.163491    0.000000      [0.163491, 0.163491]
Brier (labelset)                               0.562013    0.000000      [0.562013, 0.562013]
Exact labelset coverage                        0.996815    0.000841      [0.994725, 0.998905]
Seen-labelset rate                             1.000000    0.000000      [1.000000, 1.000000]
Coverage on seen labelsets                     0.996815    0.000841      [0.994725, 0.998905]
Jaccard>=0.5 coverage                          0.996815    0.000841      [0.994725, 0.998905]
Nominal coverage                               0.900000    0.000000      [0.900000, 0.900000]
Exact coverage gap                             0.096815    0.000841      [0.094725, 0.098905]
Average prediction-region size               135.087111    2.080888  [129.917898, 140.256324]
[L4 Efficiency]
Metric                                             Mean         Std                    95% CI
Latency P50 (ms)                               0.108367    0.001436      [0.104798, 0.111935]
Latency P99 (ms)                               0.255571    0.036139      [0.165798, 0.345345]
Tokens/sample (approx.)                        8.313818    0.000000      [8.313818, 8.313818]
Trainable parameters                           0.000000    0.000000      [0.000000, 0.000000]
Index build time (s)                           0.072443    0.000000      [0.072443, 0.072443]
[L5 Decision]
Metric                                             Mean         Std                    95% CI
Reject rate                                    1.000000    0.000000      [1.000000, 1.000000]
False-reject rate                              1.000000    0.000000      [1.000000, 1.000000]
Clarification rate                             0.000000    0.000000      [0.000000, 0.000000]
Mean clarification rounds                      0.000000    0.000000      [0.000000, 0.000000]
Set-level Jaccard error rate                   0.370667    0.000000      [0.370667, 0.370667]
Reject rate @ 95% OOS recall                   0.693636    0.000000      [0.693636, 0.693636]
False reject @ 95% OOS recall                  0.636889    0.000000      [0.636889, 0.636889]
Realized OOS recall                            0.949000    0.000000      [0.949000, 0.949000]
Tuned OOS threshold                            0.343103    0.000000      [0.343103, 0.343103]
====================================================================================================
Artifacts: D:\Project\PythonProject\www2027-intent\mulaconf-reproduction\results\mulaconf_all
```



## MixATIS

```
(mulaconf-reproduction) PS D:\Project\PythonProject\www2027-intent\mulaconf-reproduction> uv run python .\scripts\run_mulaconf.py --config .\configs\mulaconf_all.json --dataset mixatis
====================================================================================================
Multi-label conformal results | smoke=false | seeds=[42, 43, 44]
====================================================================================================

Dataset: mixatis | native multi-intent labelsets
----------------------------------------------------------------------------------------------------
Samples: atomic_labels=17, observed_labelsets=831, oos_test=0, oos_val=0, test=828, train=13162, val=759

====================================================================================================
Method: powerset
====================================================================================================

Seed 42
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.056763
  Macro-F1                                            0.387988
  Intent-set IoU                                      0.346618
  Hit@1                                               0.733092
  Hit@3                                               0.939614
  Hit@5                                               0.984300
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.051616
  Brier (labelset)                                    0.864082
  Exact labelset coverage                             0.743961
  Seen-labelset rate                                  0.868357
  Coverage on seen labelsets                          0.856745
  Jaccard>=0.5 coverage                               0.956522
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.156039
  Average prediction-region size                    142.020531
[L4 Efficiency]
  Latency P50 (ms)                                    1.304000
  Latency P99 (ms)                                    2.647201
  Tokens/sample (approx.)                            20.140097
  Trainable parameters                                       0
  Index build time (s)                                0.165870
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.663043
  Clarification-trigger accuracy                         N/A

Seed 43
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.056763
  Macro-F1                                            0.387988
  Intent-set IoU                                      0.346618
  Hit@1                                               0.733092
  Hit@3                                               0.939614
  Hit@5                                               0.984300
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.051616
  Brier (labelset)                                    0.864082
  Exact labelset coverage                             0.766908
  Seen-labelset rate                                  0.868357
  Coverage on seen labelsets                          0.883171
  Jaccard>=0.5 coverage                               0.964976
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.133092
  Average prediction-region size                    154.932367
[L4 Efficiency]
  Latency P50 (ms)                                    1.238750
  Latency P99 (ms)                                    2.211283
  Tokens/sample (approx.)                            20.140097
  Trainable parameters                                       0
  Index build time (s)                                0.165870
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.663043
  Clarification-trigger accuracy                         N/A

Seed 44
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.056763
  Macro-F1                                            0.387988
  Intent-set IoU                                      0.346618
  Hit@1                                               0.733092
  Hit@3                                               0.939614
  Hit@5                                               0.984300
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.051616
  Brier (labelset)                                    0.864082
  Exact labelset coverage                             0.766908
  Seen-labelset rate                                  0.868357
  Coverage on seen labelsets                          0.883171
  Jaccard>=0.5 coverage                               0.964976
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.133092
  Average prediction-region size                    154.932367
[L4 Efficiency]
  Latency P50 (ms)                                    1.229150
  Latency P99 (ms)                                    2.336771
  Tokens/sample (approx.)                            20.140097
  Trainable parameters                                       0
  Index build time (s)                                0.165870
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.663043
  Clarification-trigger accuracy                         N/A

Final aggregate across 3 seed(s)
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
Metric                                             Mean         Std                    95% CI
Exact Match                                    0.056763    0.000000      [0.056763, 0.056763]
Macro-F1                                       0.387988    0.000000      [0.387988, 0.387988]
Intent-set IoU                                 0.346618    0.000000      [0.346618, 0.346618]
Hit@1                                          0.733092    0.000000      [0.733092, 0.733092]
Hit@3                                          0.939614    0.000000      [0.939614, 0.939614]
Hit@5                                          0.984300    0.000000      [0.984300, 0.984300]
[L3 Calibration]
Metric                                             Mean         Std                    95% CI
ECE (15 bins)                                  0.051616    0.000000      [0.051616, 0.051616]
Brier (labelset)                               0.864082    0.000000      [0.864082, 0.864082]
Exact labelset coverage                        0.759259    0.013248      [0.726348, 0.792170]
Seen-labelset rate                             0.868357    0.000000      [0.868357, 0.868357]
Coverage on seen labelsets                     0.874363    0.015257      [0.836462, 0.912263]
Jaccard>=0.5 coverage                          0.962158    0.004881      [0.950033, 0.974283]
Nominal coverage                               0.900000    0.000000      [0.900000, 0.900000]
Exact coverage gap                            -0.140741    0.013248    [-0.173652, -0.107830]
Average prediction-region size               150.628422    7.454652  [132.110040, 169.146804]
[L4 Efficiency]
Metric                                             Mean         Std                    95% CI
Latency P50 (ms)                               1.257300    0.040727      [1.156128, 1.358472]
Latency P99 (ms)                               2.398418    0.224402      [1.840972, 2.955865]
Tokens/sample (approx.)                       20.140097    0.000000    [20.140097, 20.140097]
Trainable parameters                           0.000000    0.000000      [0.000000, 0.000000]
Index build time (s)                           0.165870    0.000000      [0.165870, 0.165870]
[L5 Decision]
Metric                                             Mean         Std                    95% CI
Reject rate                                    1.000000    0.000000      [1.000000, 1.000000]
False-reject rate                              1.000000    0.000000      [1.000000, 1.000000]
Clarification rate                             0.000000    0.000000      [0.000000, 0.000000]
Mean clarification rounds                      0.000000    0.000000      [0.000000, 0.000000]
Set-level Jaccard error rate                   0.663043    0.000000      [0.663043, 0.663043]

====================================================================================================
Method: tree_fixed_alpha
====================================================================================================

Seed 42
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.056763
  Macro-F1                                            0.387988
  Intent-set IoU                                      0.346618
  Hit@1                                               0.733092
  Hit@3                                               0.939614
  Hit@5                                               0.984300
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.051616
  Brier (labelset)                                    0.864082
  Exact labelset coverage                             0.855072
  Seen-labelset rate                                  0.868357
  Coverage on seen labelsets                          0.984701
  Jaccard>=0.5 coverage                               0.997585
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.044928
  Average prediction-region size                    750.298309
[L4 Efficiency]
  Latency P50 (ms)                                    1.304000
  Latency P99 (ms)                                    2.647201
  Tokens/sample (approx.)                            20.140097
  Trainable parameters                                       0
  Index build time (s)                                0.165870
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.663043
  Clarification-trigger accuracy                         N/A

Seed 43
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.056763
  Macro-F1                                            0.387988
  Intent-set IoU                                      0.346618
  Hit@1                                               0.733092
  Hit@3                                               0.939614
  Hit@5                                               0.984300
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.051616
  Brier (labelset)                                    0.864082
  Exact labelset coverage                             0.857488
  Seen-labelset rate                                  0.868357
  Coverage on seen labelsets                          0.987483
  Jaccard>=0.5 coverage                               0.997585
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.042512
  Average prediction-region size                    759.684783
[L4 Efficiency]
  Latency P50 (ms)                                    1.238750
  Latency P99 (ms)                                    2.211283
  Tokens/sample (approx.)                            20.140097
  Trainable parameters                                       0
  Index build time (s)                                0.165870
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.663043
  Clarification-trigger accuracy                         N/A

Seed 44
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.056763
  Macro-F1                                            0.387988
  Intent-set IoU                                      0.346618
  Hit@1                                               0.733092
  Hit@3                                               0.939614
  Hit@5                                               0.984300
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.051616
  Brier (labelset)                                    0.864082
  Exact labelset coverage                             0.847826
  Seen-labelset rate                                  0.868357
  Coverage on seen labelsets                          0.976356
  Jaccard>=0.5 coverage                               0.997585
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.052174
  Average prediction-region size                    735.429952
[L4 Efficiency]
  Latency P50 (ms)                                    1.229150
  Latency P99 (ms)                                    2.336771
  Tokens/sample (approx.)                            20.140097
  Trainable parameters                                       0
  Index build time (s)                                0.165870
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.663043
  Clarification-trigger accuracy                         N/A

Final aggregate across 3 seed(s)
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
Metric                                             Mean         Std                    95% CI
Exact Match                                    0.056763    0.000000      [0.056763, 0.056763]
Macro-F1                                       0.387988    0.000000      [0.387988, 0.387988]
Intent-set IoU                                 0.346618    0.000000      [0.346618, 0.346618]
Hit@1                                          0.733092    0.000000      [0.733092, 0.733092]
Hit@3                                          0.939614    0.000000      [0.939614, 0.939614]
Hit@5                                          0.984300    0.000000      [0.984300, 0.984300]
[L3 Calibration]
Metric                                             Mean         Std                    95% CI
ECE (15 bins)                                  0.051616    0.000000      [0.051616, 0.051616]
Brier (labelset)                               0.864082    0.000000      [0.864082, 0.864082]
Exact labelset coverage                        0.853462    0.005028      [0.840971, 0.865953]
Seen-labelset rate                             0.868357    0.000000      [0.868357, 0.868357]
Coverage on seen labelsets                     0.982847    0.005790      [0.968462, 0.997231]
Jaccard>=0.5 coverage                          0.997585    0.000000      [0.997585, 0.997585]
Nominal coverage                               0.900000    0.000000      [0.900000, 0.900000]
Exact coverage gap                            -0.046538    0.005028    [-0.059029, -0.034047]
Average prediction-region size               748.471014   12.230227  [718.089446, 778.852583]
[L4 Efficiency]
Metric                                             Mean         Std                    95% CI
Latency P50 (ms)                               1.257300    0.040727      [1.156128, 1.358472]
Latency P99 (ms)                               2.398418    0.224402      [1.840972, 2.955865]
Tokens/sample (approx.)                       20.140097    0.000000    [20.140097, 20.140097]
Trainable parameters                           0.000000    0.000000      [0.000000, 0.000000]
Index build time (s)                           0.165870    0.000000      [0.165870, 0.165870]
[L5 Decision]
Metric                                             Mean         Std                    95% CI
Reject rate                                    1.000000    0.000000      [1.000000, 1.000000]
False-reject rate                              1.000000    0.000000      [1.000000, 1.000000]
Clarification rate                             0.000000    0.000000      [0.000000, 0.000000]
Mean clarification rounds                      0.000000    0.000000      [0.000000, 0.000000]
Set-level Jaccard error rate                   0.663043    0.000000      [0.663043, 0.663043]
====================================================================================================
Artifacts: D:\Project\PythonProject\www2027-intent\mulaconf-reproduction\results\mulaconf_all
```





## MixSNIPS

```
(mulaconf-reproduction) PS D:\Project\PythonProject\www2027-intent\mulaconf-reproduction> uv run python .\scripts\run_mulaconf.py --config .\configs\mulaconf_all.json --dataset mixsnips
====================================================================================================
Multi-label conformal results | smoke=false | seeds=[42, 43, 44]
====================================================================================================

Dataset: mixsnips | native multi-intent labelsets
----------------------------------------------------------------------------------------------------
Samples: atomic_labels=7, observed_labelsets=63, oos_test=0, oos_val=0, test=2199, train=39776, val=2198

====================================================================================================
Method: powerset
====================================================================================================

Seed 42
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.559345
  Macro-F1                                            0.851229
  Intent-set IoU                                      0.793876
  Hit@1                                               0.969532
  Hit@3                                               0.996817
  Hit@5                                               0.999545
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.536618
  Brier (labelset)                                    0.972026
  Exact labelset coverage                             0.871760
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.871760
  Jaccard>=0.5 coverage                               0.930878
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.028240
  Average prediction-region size                      9.639382
[L4 Efficiency]
  Latency P50 (ms)                                    0.138700
  Latency P99 (ms)                                    0.328802
  Tokens/sample (approx.)                            19.723511
  Trainable parameters                                       0
  Index build time (s)                                0.480848
[L5 Decision]
  Reject rate                                         0.806276
  False-reject rate                                   0.806276
  Clarification rate                                  0.155525
  Clarification Jaccard coverage                      0.979532
  Mean clarification rounds                           0.155525
  Set-level Jaccard error rate                        0.113688
  Clarification-trigger accuracy                         N/A

Seed 43
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.559345
  Macro-F1                                            0.851229
  Intent-set IoU                                      0.793876
  Hit@1                                               0.969532
  Hit@3                                               0.996817
  Hit@5                                               0.999545
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.536618
  Brier (labelset)                                    0.972026
  Exact labelset coverage                             0.890405
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.890405
  Jaccard>=0.5 coverage                               0.947249
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.009595
  Average prediction-region size                     10.467030
[L4 Efficiency]
  Latency P50 (ms)                                    0.143600
  Latency P99 (ms)                                    0.541674
  Tokens/sample (approx.)                            19.723511
  Trainable parameters                                       0
  Index build time (s)                                0.480848
[L5 Decision]
  Reject rate                                         0.844475
  False-reject rate                                   0.844475
  Clarification rate                                  0.129150
  Clarification Jaccard coverage                      0.961268
  Mean clarification rounds                           0.129150
  Set-level Jaccard error rate                        0.113688
  Clarification-trigger accuracy                         N/A

Seed 44
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.559345
  Macro-F1                                            0.851229
  Intent-set IoU                                      0.793876
  Hit@1                                               0.969532
  Hit@3                                               0.996817
  Hit@5                                               0.999545
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.536618
  Brier (labelset)                                    0.972026
  Exact labelset coverage                             0.871760
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.871760
  Jaccard>=0.5 coverage                               0.930878
  Nominal coverage                                    0.900000
  Exact coverage gap                                 -0.028240
  Average prediction-region size                      9.639382
[L4 Efficiency]
  Latency P50 (ms)                                    0.147000
  Latency P99 (ms)                                    0.395410
  Tokens/sample (approx.)                            19.723511
  Trainable parameters                                       0
  Index build time (s)                                0.480848
[L5 Decision]
  Reject rate                                         0.806276
  False-reject rate                                   0.806276
  Clarification rate                                  0.155525
  Clarification Jaccard coverage                      0.979532
  Mean clarification rounds                           0.155525
  Set-level Jaccard error rate                        0.113688
  Clarification-trigger accuracy                         N/A

Final aggregate across 3 seed(s)
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
Metric                                             Mean         Std                    95% CI
Exact Match                                    0.559345    0.000000      [0.559345, 0.559345]
Macro-F1                                       0.851229    0.000000      [0.851229, 0.851229]
Intent-set IoU                                 0.793876    0.000000      [0.793876, 0.793876]
Hit@1                                          0.969532    0.000000      [0.969532, 0.969532]
Hit@3                                          0.996817    0.000000      [0.996817, 0.996817]
Hit@5                                          0.999545    0.000000      [0.999545, 0.999545]
[L3 Calibration]
Metric                                             Mean         Std                    95% CI
ECE (15 bins)                                  0.536618    0.000000      [0.536618, 0.536618]
Brier (labelset)                               0.972026    0.000000      [0.972026, 0.972026]
Exact labelset coverage                        0.877975    0.010765      [0.851234, 0.904716]
Seen-labelset rate                             1.000000    0.000000      [1.000000, 1.000000]
Coverage on seen labelsets                     0.877975    0.010765      [0.851234, 0.904716]
Jaccard>=0.5 coverage                          0.936335    0.009452      [0.912855, 0.959814]
Nominal coverage                               0.900000    0.000000      [0.900000, 0.900000]
Exact coverage gap                            -0.022025    0.010765     [-0.048766, 0.004716]
Average prediction-region size                 9.915265    0.477843     [8.728236, 11.102293]
[L4 Efficiency]
Metric                                             Mean         Std                    95% CI
Latency P50 (ms)                               0.143100    0.004173      [0.132735, 0.153465]
Latency P99 (ms)                               0.421962    0.108892      [0.151460, 0.692464]
Tokens/sample (approx.)                       19.723511    0.000000    [19.723511, 19.723511]
Trainable parameters                           0.000000    0.000000      [0.000000, 0.000000]
Index build time (s)                           0.480848    0.000000      [0.480848, 0.480848]
[L5 Decision]
Metric                                             Mean         Std                    95% CI
Reject rate                                    0.819009    0.022054      [0.764223, 0.873795]
False-reject rate                              0.819009    0.022054      [0.764223, 0.873795]
Clarification rate                             0.146733    0.015228      [0.108905, 0.184562]
Clarification Jaccard coverage                 0.973444    0.010545      [0.947249, 0.999639]
Mean clarification rounds                      0.146733    0.015228      [0.108905, 0.184562]
Set-level Jaccard error rate                   0.113688    0.000000      [0.113688, 0.113688]

====================================================================================================
Method: tree_fixed_alpha
====================================================================================================

Seed 42
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.559345
  Macro-F1                                            0.851229
  Intent-set IoU                                      0.793876
  Hit@1                                               0.969532
  Hit@3                                               0.996817
  Hit@5                                               0.999545
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.536618
  Brier (labelset)                                    0.972026
  Exact labelset coverage                             0.935880
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.935880
  Jaccard>=0.5 coverage                               0.996817
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.035880
  Average prediction-region size                     24.828558
[L4 Efficiency]
  Latency P50 (ms)                                    0.138700
  Latency P99 (ms)                                    0.328802
  Tokens/sample (approx.)                            19.723511
  Trainable parameters                                       0
  Index build time (s)                                0.480848
[L5 Decision]
  Reject rate                                         0.999545
  False-reject rate                                   0.999545
  Clarification rate                                  0.000455
  Clarification Jaccard coverage                             1
  Mean clarification rounds                           0.000455
  Set-level Jaccard error rate                        0.113688
  Clarification-trigger accuracy                         N/A

Seed 43
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.559345
  Macro-F1                                            0.851229
  Intent-set IoU                                      0.793876
  Hit@1                                               0.969532
  Hit@3                                               0.996817
  Hit@5                                               0.999545
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.536618
  Brier (labelset)                                    0.972026
  Exact labelset coverage                             0.949977
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.949977
  Jaccard>=0.5 coverage                               0.997271
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.049977
  Average prediction-region size                     26.224193
[L4 Efficiency]
  Latency P50 (ms)                                    0.143600
  Latency P99 (ms)                                    0.541674
  Tokens/sample (approx.)                            19.723511
  Trainable parameters                                       0
  Index build time (s)                                0.480848
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.113688
  Clarification-trigger accuracy                         N/A

Seed 44
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
  Exact Match                                         0.559345
  Macro-F1                                            0.851229
  Intent-set IoU                                      0.793876
  Hit@1                                               0.969532
  Hit@3                                               0.996817
  Hit@5                                               0.999545
[L2 OOS]
  Not evaluated: dataset has no OOS split
[L3 Calibration]
  ECE (15 bins)                                       0.536618
  Brier (labelset)                                    0.972026
  Exact labelset coverage                             0.949523
  Seen-labelset rate                                         1
  Coverage on seen labelsets                          0.949523
  Jaccard>=0.5 coverage                               0.996817
  Nominal coverage                                    0.900000
  Exact coverage gap                                  0.049523
  Average prediction-region size                     25.201000
[L4 Efficiency]
  Latency P50 (ms)                                    0.147000
  Latency P99 (ms)                                    0.395410
  Tokens/sample (approx.)                            19.723511
  Trainable parameters                                       0
  Index build time (s)                                0.480848
[L5 Decision]
  Reject rate                                                1
  False-reject rate                                          1
  Clarification rate                                  0.000000
  Mean clarification rounds                           0.000000
  Set-level Jaccard error rate                        0.113688
  Clarification-trigger accuracy                         N/A

Final aggregate across 3 seed(s)
----------------------------------------------------------------------------------------------------
[L1 Multi-intent]
Metric                                             Mean         Std                    95% CI
Exact Match                                    0.559345    0.000000      [0.559345, 0.559345]
Macro-F1                                       0.851229    0.000000      [0.851229, 0.851229]
Intent-set IoU                                 0.793876    0.000000      [0.793876, 0.793876]
Hit@1                                          0.969532    0.000000      [0.969532, 0.969532]
Hit@3                                          0.996817    0.000000      [0.996817, 0.996817]
Hit@5                                          0.999545    0.000000      [0.999545, 0.999545]
[L3 Calibration]
Metric                                             Mean         Std                    95% CI
ECE (15 bins)                                  0.536618    0.000000      [0.536618, 0.536618]
Brier (labelset)                               0.972026    0.000000      [0.972026, 0.972026]
Exact labelset coverage                        0.945127    0.008011      [0.925226, 0.965027]
Seen-labelset rate                             1.000000    0.000000      [1.000000, 1.000000]
Coverage on seen labelsets                     0.945127    0.008011      [0.925226, 0.965027]
Jaccard>=0.5 coverage                          0.996968    0.000263      [0.996316, 0.997621]
Nominal coverage                               0.900000    0.000000      [0.900000, 0.900000]
Exact coverage gap                             0.045127    0.008011      [0.025226, 0.065027]
Average prediction-region size                25.417917    0.722661    [23.622729, 27.213106]
[L4 Efficiency]
Metric                                             Mean         Std                    95% CI
Latency P50 (ms)                               0.143100    0.004173      [0.132735, 0.153465]
Latency P99 (ms)                               0.421962    0.108892      [0.151460, 0.692464]
Tokens/sample (approx.)                       19.723511    0.000000    [19.723511, 19.723511]
Trainable parameters                           0.000000    0.000000      [0.000000, 0.000000]
Index build time (s)                           0.480848    0.000000      [0.480848, 0.480848]
[L5 Decision]
Metric                                             Mean         Std                    95% CI
Reject rate                                    0.999848    0.000263      [0.999196, 1.000501]
False-reject rate                              0.999848    0.000263      [0.999196, 1.000501]
Clarification rate                             0.000152    0.000263     [-0.000501, 0.000804]
Clarification Jaccard coverage                 1.000000    0.000000      [1.000000, 1.000000]
Mean clarification rounds                      0.000152    0.000263     [-0.000501, 0.000804]
Set-level Jaccard error rate                   0.113688    0.000000      [0.113688, 0.113688]
====================================================================================================
Artifacts: D:\Project\PythonProject\www2027-intent\mulaconf-reproduction\results\mulaconf_all
```

