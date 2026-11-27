| model | test | accuracy | precision | recall | f1 | auroc | fpr | status |
|---|---|---|---|---|---|---|---|---|
| Chance baseline | test_sd | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | analytic |
| Chance baseline | test_mj | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | analytic |
| Chance baseline | test_mj_native | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | analytic |
| Custom CNN | test_sd | 0.8970 | 0.8970 | 0.8970 | 0.8970 | 0.9613 | 0.1030 | evaluated |
| Custom CNN | test_mj | 0.5140 | 0.5598 | 0.1310 | 0.2123 | 0.4646 | 0.1030 | evaluated |
| Custom CNN | test_mj_native | 0.5112 | 0.5470 | 0.1310 | 0.2114 | 0.4690 | 0.1085 | evaluated |
| ResNet-18 | test_sd | 0.9633 | 0.9598 | 0.9670 | 0.9634 | 0.9940 | 0.0405 | evaluated |
| ResNet-18 | test_mj | 0.6082 | 0.8639 | 0.2570 | 0.3961 | 0.7272 | 0.0405 | evaluated |
| ResNet-18 | test_mj_native | 0.6035 | 0.8371 | 0.2570 | 0.3933 | 0.7215 | 0.0500 | evaluated |
