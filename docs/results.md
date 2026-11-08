# 消融实验结果

数据集：HRSC2016（单类 ship），12 epoch。

| 组名 | CR Loss | APS | DRoI | mAP | Recall | 用时(分钟) |
|---|---|---|---|---|---|---|
| 000_baseline | False | False | False | 0.7866 | - | 80.1 |
| 001_droi_only | False | False | True | 0.7856 | - | 76.4 |
| 010_aps_only | False | True | False | 0.7911 | - | 145.8 |
| 011_aps_droi | False | True | True | 0.7792 | - | 145.1 |
| 100_crloss_only | - | - | - | TBD | - | - |
| 101_crloss_droi | - | - | - | TBD | - | - |
| 110_crloss_aps | - | - | - | TBD | - | - |
| 111_full | - | - | - | TBD | - | - |

## 说明

- 原论文数据集未公开，本仓库用 HRSC2016 替代。
- 绝对值不能与论文直接对比，关注相对趋势。
