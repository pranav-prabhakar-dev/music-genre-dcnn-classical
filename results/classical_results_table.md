| Features | split | LR | GDA | RF | SVM |
|---|---|---|---|---|---|
| Raw MFCC (25800) | train | 1.000 | 0.890 | 1.000 | 0.993 |
| Raw MFCC (25800) | test | 0.840 | 0.830 | 0.740 | 0.790 |
| PCA of raw (50) | train | 1.000 | 0.863 | 1.000 | 0.932 |
| PCA of raw (50) | test | 0.850 | 0.790 | 0.820 | 0.830 |
| DCNN layer 1 (320) | train | 1.000 | 1.000 | 0.998 | 0.973 |
| DCNN layer 1 (320) | test | 0.880 | 0.610 | 0.870 | 0.880 |
| DCNN layer 2 (48) | train | 0.998 | 0.988 | 0.998 | 0.988 |
| DCNN layer 2 (48) | test | 0.890 | 0.860 | 0.870 | 0.860 |

DCNN alone: train 0.990, test 0.890
