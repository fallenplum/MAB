# B1 已知总体合成实验

## 研究问题

冻结评估系统后，检查真实总体收益估计、区间覆盖、错误批准和正收益检出能力。每个数值由2000批独立合成问题簇及购买随机化产生。

## 实验设置与对照

六类总体、200/1000/5000题、每题三个生成重复。主实验题内条件独立，敏感性实验题内完全相关。辅助模型仅用独立历史600题。ASI和Robust为明确标注的B1采样适配，使用共同AIPW估计与推断。

预算为总体期望Gold与执行费用各25%。当前评价输入按已知总体类别质量制定固定概率函数。实际开销可以超过期望预算。

## 数值结果

| 环境 | 题数 | 相关性 | 方法 | RMSE | 渐近覆盖 | 保守覆盖 | 保守采用率 | 渐近采用率 | 平均Gold | 平均Token |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| null_zero | 200 | conditionally_independent | U0 | 0.03653 | 0.9485 | 1.0000 | 0.0000 | 0.0475 | 50.06 | 400564 |
| null_zero | 200 | conditionally_independent | U1 | 0.03659 | 0.9485 | 1.0000 | 0.0000 | 0.0465 | 50.06 | 400564 |
| null_zero | 200 | conditionally_independent | ASI-adapted | 0.03421 | 0.9465 | 1.0000 | 0.0000 | 0.0510 | 50.14 | 401063 |
| null_zero | 200 | conditionally_independent | Robust-ASI-adapted | 0.03494 | 0.9480 | 1.0000 | 0.0000 | 0.0560 | 50.04 | 400290 |
| null_zero | 200 | conditionally_independent | T0 | 0.03419 | 0.9465 | 1.0000 | 0.0000 | 0.0510 | 50.14 | 401096 |
| null_zero | 200 | conditionally_independent | T1 | 0.03414 | 0.9465 | 1.0000 | 0.0000 | 0.0515 | 50.14 | 401064 |
| null_zero | 200 | perfectly_correlated | U0 | 0.06271 | 0.9580 | 1.0000 | 0.0000 | 0.0530 | 50.13 | 400817 |
| null_zero | 200 | perfectly_correlated | U1 | 0.06276 | 0.9505 | 1.0000 | 0.0000 | 0.0495 | 50.13 | 400817 |
| null_zero | 200 | perfectly_correlated | ASI-adapted | 0.05717 | 0.9615 | 1.0000 | 0.0000 | 0.0455 | 50.12 | 400751 |
| null_zero | 200 | perfectly_correlated | Robust-ASI-adapted | 0.05805 | 0.9565 | 1.0000 | 0.0000 | 0.0445 | 50.13 | 400813 |
| null_zero | 200 | perfectly_correlated | T0 | 0.05720 | 0.9620 | 1.0000 | 0.0000 | 0.0460 | 50.13 | 400816 |
| null_zero | 200 | perfectly_correlated | T1 | 0.05723 | 0.9605 | 1.0000 | 0.0000 | 0.0465 | 50.12 | 400759 |
| null_zero | 1000 | conditionally_independent | U0 | 0.01654 | 0.9470 | 1.0000 | 0.0000 | 0.0475 | 249.59 | 1997172 |
| null_zero | 1000 | conditionally_independent | U1 | 0.01665 | 0.9435 | 1.0000 | 0.0000 | 0.0445 | 249.59 | 1997172 |
| null_zero | 1000 | conditionally_independent | ASI-adapted | 0.01531 | 0.9470 | 1.0000 | 0.0000 | 0.0460 | 249.57 | 1996496 |
| null_zero | 1000 | conditionally_independent | Robust-ASI-adapted | 0.01551 | 0.9495 | 1.0000 | 0.0000 | 0.0450 | 249.59 | 1996733 |
| null_zero | 1000 | conditionally_independent | T0 | 0.01531 | 0.9475 | 1.0000 | 0.0000 | 0.0460 | 249.55 | 1996357 |
| null_zero | 1000 | conditionally_independent | T1 | 0.01528 | 0.9480 | 1.0000 | 0.0000 | 0.0450 | 249.61 | 1996826 |
| null_zero | 1000 | perfectly_correlated | U0 | 0.02788 | 0.9520 | 1.0000 | 0.0000 | 0.0430 | 250.25 | 2002034 |
| null_zero | 1000 | perfectly_correlated | U1 | 0.02794 | 0.9505 | 1.0000 | 0.0000 | 0.0430 | 250.25 | 2002034 |
| null_zero | 1000 | perfectly_correlated | ASI-adapted | 0.02599 | 0.9440 | 1.0000 | 0.0000 | 0.0485 | 250.02 | 1999837 |
| null_zero | 1000 | perfectly_correlated | Robust-ASI-adapted | 0.02629 | 0.9505 | 1.0000 | 0.0000 | 0.0465 | 250.24 | 2001761 |
| null_zero | 1000 | perfectly_correlated | T0 | 0.02600 | 0.9440 | 1.0000 | 0.0000 | 0.0480 | 250.00 | 1999744 |
| null_zero | 1000 | perfectly_correlated | T1 | 0.02600 | 0.9455 | 1.0000 | 0.0000 | 0.0485 | 250.01 | 1999816 |
| null_zero | 5000 | conditionally_independent | U0 | 0.00724 | 0.9510 | 1.0000 | 0.0000 | 0.0495 | 1248.30 | 9985398 |
| null_zero | 5000 | conditionally_independent | U1 | 0.00725 | 0.9510 | 1.0000 | 0.0000 | 0.0500 | 1248.30 | 9985398 |
| null_zero | 5000 | conditionally_independent | ASI-adapted | 0.00675 | 0.9500 | 1.0000 | 0.0000 | 0.0495 | 1248.47 | 9986859 |
| null_zero | 5000 | conditionally_independent | Robust-ASI-adapted | 0.00689 | 0.9475 | 1.0000 | 0.0000 | 0.0485 | 1248.57 | 9987582 |
| null_zero | 5000 | conditionally_independent | T0 | 0.00675 | 0.9500 | 1.0000 | 0.0000 | 0.0495 | 1248.46 | 9986810 |
| null_zero | 5000 | conditionally_independent | T1 | 0.00674 | 0.9510 | 1.0000 | 0.0000 | 0.0510 | 1248.51 | 9987269 |
| null_zero | 5000 | perfectly_correlated | U0 | 0.01259 | 0.9435 | 1.0000 | 0.0005 | 0.0540 | 1248.79 | 9990590 |
| null_zero | 5000 | perfectly_correlated | U1 | 0.01260 | 0.9425 | 1.0000 | 0.0005 | 0.0535 | 1248.79 | 9990590 |
| null_zero | 5000 | perfectly_correlated | ASI-adapted | 0.01153 | 0.9530 | 1.0000 | 0.0000 | 0.0425 | 1248.70 | 9990422 |
| null_zero | 5000 | perfectly_correlated | Robust-ASI-adapted | 0.01162 | 0.9540 | 1.0000 | 0.0005 | 0.0440 | 1248.26 | 9986407 |
| null_zero | 5000 | perfectly_correlated | T0 | 0.01152 | 0.9540 | 1.0000 | 0.0000 | 0.0435 | 1248.66 | 9990077 |
| null_zero | 5000 | perfectly_correlated | T1 | 0.01150 | 0.9550 | 1.0000 | 0.0000 | 0.0420 | 1248.62 | 9989737 |
| negative_2pp | 200 | conditionally_independent | U0 | 0.03618 | 0.9555 | 1.0000 | 0.0000 | 0.0115 | 49.96 | 399467 |
| negative_2pp | 200 | conditionally_independent | U1 | 0.03615 | 0.9550 | 1.0000 | 0.0000 | 0.0120 | 49.96 | 399467 |
| negative_2pp | 200 | conditionally_independent | ASI-adapted | 0.03356 | 0.9490 | 1.0000 | 0.0000 | 0.0080 | 49.96 | 399476 |
| negative_2pp | 200 | conditionally_independent | Robust-ASI-adapted | 0.03410 | 0.9510 | 1.0000 | 0.0000 | 0.0090 | 49.96 | 399426 |
| negative_2pp | 200 | conditionally_independent | T0 | 0.03356 | 0.9510 | 1.0000 | 0.0000 | 0.0080 | 49.96 | 399454 |
| negative_2pp | 200 | conditionally_independent | T1 | 0.03359 | 0.9505 | 1.0000 | 0.0000 | 0.0080 | 49.96 | 399488 |
| negative_2pp | 200 | perfectly_correlated | U0 | 0.06340 | 0.9460 | 1.0000 | 0.0000 | 0.0300 | 50.07 | 400399 |
| negative_2pp | 200 | perfectly_correlated | U1 | 0.06336 | 0.9460 | 1.0000 | 0.0000 | 0.0300 | 50.07 | 400399 |
| negative_2pp | 200 | perfectly_correlated | ASI-adapted | 0.05784 | 0.9555 | 1.0000 | 0.0000 | 0.0230 | 49.96 | 399525 |
| negative_2pp | 200 | perfectly_correlated | Robust-ASI-adapted | 0.05911 | 0.9565 | 1.0000 | 0.0000 | 0.0240 | 49.98 | 399680 |
| negative_2pp | 200 | perfectly_correlated | T0 | 0.05796 | 0.9550 | 1.0000 | 0.0000 | 0.0245 | 49.96 | 399546 |
| negative_2pp | 200 | perfectly_correlated | T1 | 0.05792 | 0.9550 | 1.0000 | 0.0000 | 0.0245 | 49.95 | 399450 |
| negative_2pp | 1000 | conditionally_independent | U0 | 0.01628 | 0.9510 | 1.0000 | 0.0000 | 0.0025 | 250.01 | 2000040 |
| negative_2pp | 1000 | conditionally_independent | U1 | 0.01628 | 0.9530 | 1.0000 | 0.0000 | 0.0025 | 250.01 | 2000040 |
| negative_2pp | 1000 | conditionally_independent | ASI-adapted | 0.01532 | 0.9435 | 1.0000 | 0.0000 | 0.0015 | 249.82 | 1998529 |
| negative_2pp | 1000 | conditionally_independent | Robust-ASI-adapted | 0.01545 | 0.9445 | 1.0000 | 0.0000 | 0.0010 | 249.90 | 1999267 |
| negative_2pp | 1000 | conditionally_independent | T0 | 0.01531 | 0.9425 | 1.0000 | 0.0000 | 0.0010 | 249.83 | 1998654 |
| negative_2pp | 1000 | conditionally_independent | T1 | 0.01533 | 0.9420 | 1.0000 | 0.0000 | 0.0010 | 249.83 | 1998669 |
| negative_2pp | 1000 | perfectly_correlated | U0 | 0.02844 | 0.9470 | 1.0000 | 0.0000 | 0.0075 | 249.87 | 1999212 |
| negative_2pp | 1000 | perfectly_correlated | U1 | 0.02846 | 0.9495 | 1.0000 | 0.0000 | 0.0085 | 249.87 | 1999212 |
| negative_2pp | 1000 | perfectly_correlated | ASI-adapted | 0.02620 | 0.9570 | 1.0000 | 0.0000 | 0.0105 | 250.03 | 2000106 |
| negative_2pp | 1000 | perfectly_correlated | Robust-ASI-adapted | 0.02629 | 0.9475 | 1.0000 | 0.0000 | 0.0045 | 250.20 | 2001640 |
| negative_2pp | 1000 | perfectly_correlated | T0 | 0.02625 | 0.9535 | 1.0000 | 0.0000 | 0.0100 | 250.05 | 2000301 |
| negative_2pp | 1000 | perfectly_correlated | T1 | 0.02623 | 0.9550 | 1.0000 | 0.0000 | 0.0100 | 250.04 | 2000162 |
| negative_2pp | 5000 | conditionally_independent | U0 | 0.00729 | 0.9555 | 1.0000 | 0.0000 | 0.0000 | 1249.01 | 9990859 |
| negative_2pp | 5000 | conditionally_independent | U1 | 0.00727 | 0.9545 | 1.0000 | 0.0000 | 0.0000 | 1249.01 | 9990859 |
| negative_2pp | 5000 | conditionally_independent | ASI-adapted | 0.00682 | 0.9495 | 1.0000 | 0.0000 | 0.0000 | 1248.70 | 9988579 |
| negative_2pp | 5000 | conditionally_independent | Robust-ASI-adapted | 0.00688 | 0.9485 | 1.0000 | 0.0000 | 0.0000 | 1248.38 | 9986069 |
| negative_2pp | 5000 | conditionally_independent | T0 | 0.00681 | 0.9500 | 1.0000 | 0.0000 | 0.0000 | 1248.68 | 9988557 |
| negative_2pp | 5000 | conditionally_independent | T1 | 0.00681 | 0.9500 | 1.0000 | 0.0000 | 0.0000 | 1248.71 | 9988744 |
| negative_2pp | 5000 | perfectly_correlated | U0 | 0.01282 | 0.9420 | 1.0000 | 0.0000 | 0.0000 | 1250.24 | 10000687 |
| negative_2pp | 5000 | perfectly_correlated | U1 | 0.01281 | 0.9410 | 1.0000 | 0.0000 | 0.0000 | 1250.24 | 10000687 |
| negative_2pp | 5000 | perfectly_correlated | ASI-adapted | 0.01193 | 0.9420 | 1.0000 | 0.0000 | 0.0000 | 1250.03 | 10000003 |
| negative_2pp | 5000 | perfectly_correlated | Robust-ASI-adapted | 0.01208 | 0.9445 | 1.0000 | 0.0000 | 0.0000 | 1250.13 | 10000329 |
| negative_2pp | 5000 | perfectly_correlated | T0 | 0.01193 | 0.9405 | 1.0000 | 0.0000 | 0.0000 | 1250.03 | 9999875 |
| negative_2pp | 5000 | perfectly_correlated | T1 | 0.01192 | 0.9405 | 1.0000 | 0.0000 | 0.0000 | 1250.00 | 9999650 |
| positive_1pp | 200 | conditionally_independent | U0 | 0.03644 | 0.9490 | 1.0000 | 0.0000 | 0.0890 | 50.14 | 401176 |
| positive_1pp | 200 | conditionally_independent | U1 | 0.03657 | 0.9495 | 1.0000 | 0.0000 | 0.0915 | 50.14 | 401176 |
| positive_1pp | 200 | conditionally_independent | ASI-adapted | 0.03327 | 0.9530 | 1.0000 | 0.0000 | 0.0890 | 50.10 | 400782 |
| positive_1pp | 200 | conditionally_independent | Robust-ASI-adapted | 0.03378 | 0.9475 | 1.0000 | 0.0000 | 0.0900 | 50.07 | 400558 |
| positive_1pp | 200 | conditionally_independent | T0 | 0.03329 | 0.9525 | 1.0000 | 0.0000 | 0.0905 | 50.09 | 400710 |
| positive_1pp | 200 | conditionally_independent | T1 | 0.03318 | 0.9535 | 1.0000 | 0.0000 | 0.0900 | 50.10 | 400779 |
| positive_1pp | 200 | perfectly_correlated | U0 | 0.06385 | 0.9415 | 1.0000 | 0.0000 | 0.0740 | 49.99 | 399948 |
| positive_1pp | 200 | perfectly_correlated | U1 | 0.06393 | 0.9460 | 1.0000 | 0.0000 | 0.0780 | 49.99 | 399948 |
| positive_1pp | 200 | perfectly_correlated | ASI-adapted | 0.05809 | 0.9565 | 1.0000 | 0.0000 | 0.0730 | 50.06 | 400510 |
| positive_1pp | 200 | perfectly_correlated | Robust-ASI-adapted | 0.05892 | 0.9515 | 1.0000 | 0.0000 | 0.0730 | 50.10 | 400794 |
| positive_1pp | 200 | perfectly_correlated | T0 | 0.05819 | 0.9565 | 1.0000 | 0.0000 | 0.0740 | 50.05 | 400472 |
| positive_1pp | 200 | perfectly_correlated | T1 | 0.05804 | 0.9575 | 1.0000 | 0.0000 | 0.0715 | 50.06 | 400498 |
| positive_1pp | 1000 | conditionally_independent | U0 | 0.01583 | 0.9530 | 1.0000 | 0.0000 | 0.1485 | 250.24 | 2002010 |
| positive_1pp | 1000 | conditionally_independent | U1 | 0.01584 | 0.9545 | 1.0000 | 0.0000 | 0.1525 | 250.24 | 2002010 |
| positive_1pp | 1000 | conditionally_independent | ASI-adapted | 0.01479 | 0.9555 | 1.0000 | 0.0000 | 0.1620 | 249.91 | 1999016 |
| positive_1pp | 1000 | conditionally_independent | Robust-ASI-adapted | 0.01500 | 0.9505 | 1.0000 | 0.0000 | 0.1570 | 249.87 | 1998706 |
| positive_1pp | 1000 | conditionally_independent | T0 | 0.01479 | 0.9560 | 1.0000 | 0.0000 | 0.1640 | 249.93 | 1999246 |
| positive_1pp | 1000 | conditionally_independent | T1 | 0.01479 | 0.9545 | 1.0000 | 0.0000 | 0.1605 | 249.92 | 1999185 |
| positive_1pp | 1000 | perfectly_correlated | U0 | 0.02834 | 0.9455 | 1.0000 | 0.0005 | 0.0970 | 249.38 | 1994501 |
| positive_1pp | 1000 | perfectly_correlated | U1 | 0.02833 | 0.9460 | 1.0000 | 0.0010 | 0.1010 | 249.38 | 1994501 |
| positive_1pp | 1000 | perfectly_correlated | ASI-adapted | 0.02603 | 0.9480 | 1.0000 | 0.0000 | 0.1000 | 249.27 | 1993509 |
| positive_1pp | 1000 | perfectly_correlated | Robust-ASI-adapted | 0.02647 | 0.9455 | 1.0000 | 0.0000 | 0.0990 | 249.45 | 1994970 |
| positive_1pp | 1000 | perfectly_correlated | T0 | 0.02601 | 0.9485 | 1.0000 | 0.0000 | 0.1005 | 249.25 | 1993382 |
| positive_1pp | 1000 | perfectly_correlated | T1 | 0.02605 | 0.9460 | 1.0000 | 0.0000 | 0.1040 | 249.24 | 1993215 |
| positive_1pp | 5000 | conditionally_independent | U0 | 0.00712 | 0.9635 | 0.9995 | 0.0175 | 0.3915 | 1251.10 | 10009229 |
| positive_1pp | 5000 | conditionally_independent | U1 | 0.00712 | 0.9640 | 0.9995 | 0.0160 | 0.3955 | 1251.10 | 10009229 |
| positive_1pp | 5000 | conditionally_independent | ASI-adapted | 0.00665 | 0.9570 | 1.0000 | 0.0070 | 0.4370 | 1251.07 | 10009373 |
| positive_1pp | 5000 | conditionally_independent | Robust-ASI-adapted | 0.00671 | 0.9600 | 1.0000 | 0.0110 | 0.4325 | 1250.98 | 10008718 |
| positive_1pp | 5000 | conditionally_independent | T0 | 0.00665 | 0.9555 | 1.0000 | 0.0070 | 0.4390 | 1251.13 | 10009854 |
| positive_1pp | 5000 | conditionally_independent | T1 | 0.00664 | 0.9575 | 1.0000 | 0.0065 | 0.4385 | 1251.13 | 10009851 |
| positive_1pp | 5000 | perfectly_correlated | U0 | 0.01269 | 0.9445 | 1.0000 | 0.0110 | 0.1985 | 1249.92 | 9999725 |
| positive_1pp | 5000 | perfectly_correlated | U1 | 0.01268 | 0.9465 | 1.0000 | 0.0110 | 0.2020 | 1249.92 | 9999725 |
| positive_1pp | 5000 | perfectly_correlated | ASI-adapted | 0.01154 | 0.9565 | 1.0000 | 0.0030 | 0.2165 | 1249.92 | 9999876 |
| positive_1pp | 5000 | perfectly_correlated | Robust-ASI-adapted | 0.01176 | 0.9480 | 1.0000 | 0.0075 | 0.2155 | 1249.39 | 9995515 |
| positive_1pp | 5000 | perfectly_correlated | T0 | 0.01154 | 0.9565 | 1.0000 | 0.0035 | 0.2170 | 1249.93 | 9999975 |
| positive_1pp | 5000 | perfectly_correlated | T1 | 0.01154 | 0.9575 | 1.0000 | 0.0030 | 0.2190 | 1249.95 | 10000172 |
| positive_3pp | 200 | conditionally_independent | U0 | 0.03537 | 0.9575 | 1.0000 | 0.0000 | 0.1910 | 50.04 | 400393 |
| positive_3pp | 200 | conditionally_independent | U1 | 0.03521 | 0.9585 | 1.0000 | 0.0000 | 0.2045 | 50.04 | 400393 |
| positive_3pp | 200 | conditionally_independent | ASI-adapted | 0.03290 | 0.9545 | 1.0000 | 0.0000 | 0.2145 | 50.08 | 400808 |
| positive_3pp | 200 | conditionally_independent | Robust-ASI-adapted | 0.03361 | 0.9520 | 1.0000 | 0.0000 | 0.2135 | 50.13 | 401190 |
| positive_3pp | 200 | conditionally_independent | T0 | 0.03291 | 0.9545 | 1.0000 | 0.0000 | 0.2135 | 50.09 | 400831 |
| positive_3pp | 200 | conditionally_independent | T1 | 0.03288 | 0.9545 | 1.0000 | 0.0000 | 0.2150 | 50.08 | 400818 |
| positive_3pp | 200 | perfectly_correlated | U0 | 0.06419 | 0.9540 | 1.0000 | 0.0000 | 0.1275 | 49.96 | 399769 |
| positive_3pp | 200 | perfectly_correlated | U1 | 0.06407 | 0.9535 | 1.0000 | 0.0000 | 0.1340 | 49.96 | 399769 |
| positive_3pp | 200 | perfectly_correlated | ASI-adapted | 0.06059 | 0.9470 | 1.0000 | 0.0000 | 0.1375 | 49.89 | 399125 |
| positive_3pp | 200 | perfectly_correlated | Robust-ASI-adapted | 0.06111 | 0.9455 | 1.0000 | 0.0000 | 0.1300 | 49.95 | 399642 |
| positive_3pp | 200 | perfectly_correlated | T0 | 0.06055 | 0.9475 | 1.0000 | 0.0000 | 0.1370 | 49.89 | 399126 |
| positive_3pp | 200 | perfectly_correlated | T1 | 0.06069 | 0.9470 | 1.0000 | 0.0000 | 0.1350 | 49.87 | 398954 |
| positive_3pp | 1000 | conditionally_independent | U0 | 0.01587 | 0.9555 | 1.0000 | 0.0065 | 0.5865 | 249.58 | 1996583 |
| positive_3pp | 1000 | conditionally_independent | U1 | 0.01580 | 0.9590 | 1.0000 | 0.0060 | 0.5920 | 249.58 | 1996583 |
| positive_3pp | 1000 | conditionally_independent | ASI-adapted | 0.01457 | 0.9585 | 1.0000 | 0.0000 | 0.6605 | 249.98 | 1999692 |
| positive_3pp | 1000 | conditionally_independent | Robust-ASI-adapted | 0.01483 | 0.9560 | 1.0000 | 0.0005 | 0.6205 | 249.89 | 1999090 |
| positive_3pp | 1000 | conditionally_independent | T0 | 0.01458 | 0.9585 | 1.0000 | 0.0000 | 0.6600 | 249.99 | 1999706 |
| positive_3pp | 1000 | conditionally_independent | T1 | 0.01458 | 0.9575 | 1.0000 | 0.0000 | 0.6585 | 249.99 | 1999729 |
| positive_3pp | 1000 | perfectly_correlated | U0 | 0.02835 | 0.9460 | 1.0000 | 0.0050 | 0.2855 | 249.60 | 1997204 |
| positive_3pp | 1000 | perfectly_correlated | U1 | 0.02834 | 0.9440 | 1.0000 | 0.0055 | 0.2895 | 249.60 | 1997204 |
| positive_3pp | 1000 | perfectly_correlated | ASI-adapted | 0.02597 | 0.9470 | 1.0000 | 0.0005 | 0.3100 | 249.65 | 1997376 |
| positive_3pp | 1000 | perfectly_correlated | Robust-ASI-adapted | 0.02627 | 0.9505 | 1.0000 | 0.0005 | 0.2985 | 249.64 | 1997315 |
| positive_3pp | 1000 | perfectly_correlated | T0 | 0.02598 | 0.9470 | 1.0000 | 0.0005 | 0.3100 | 249.67 | 1997438 |
| positive_3pp | 1000 | perfectly_correlated | T1 | 0.02595 | 0.9460 | 1.0000 | 0.0005 | 0.3110 | 249.64 | 1997275 |
| positive_3pp | 5000 | conditionally_independent | U0 | 0.00722 | 0.9535 | 0.9995 | 0.7820 | 0.9940 | 1249.79 | 9998523 |
| positive_3pp | 5000 | conditionally_independent | U1 | 0.00719 | 0.9560 | 0.9995 | 0.7780 | 0.9950 | 1249.79 | 9998523 |
| positive_3pp | 5000 | conditionally_independent | ASI-adapted | 0.00665 | 0.9545 | 1.0000 | 0.7280 | 0.9965 | 1249.62 | 9997449 |
| positive_3pp | 5000 | conditionally_independent | Robust-ASI-adapted | 0.00683 | 0.9500 | 1.0000 | 0.7805 | 0.9980 | 1249.79 | 9998974 |
| positive_3pp | 5000 | conditionally_independent | T0 | 0.00665 | 0.9555 | 1.0000 | 0.7285 | 0.9965 | 1249.56 | 9997025 |
| positive_3pp | 5000 | conditionally_independent | T1 | 0.00666 | 0.9550 | 1.0000 | 0.7270 | 0.9965 | 1249.51 | 9996597 |
| positive_3pp | 5000 | perfectly_correlated | U0 | 0.01294 | 0.9445 | 0.9990 | 0.2230 | 0.7385 | 1250.15 | 10002588 |
| positive_3pp | 5000 | perfectly_correlated | U1 | 0.01292 | 0.9425 | 0.9990 | 0.2215 | 0.7420 | 1250.15 | 10002588 |
| positive_3pp | 5000 | perfectly_correlated | ASI-adapted | 0.01183 | 0.9445 | 1.0000 | 0.1970 | 0.8085 | 1249.62 | 9998524 |
| positive_3pp | 5000 | perfectly_correlated | Robust-ASI-adapted | 0.01212 | 0.9440 | 1.0000 | 0.2275 | 0.8020 | 1250.18 | 10002509 |
| positive_3pp | 5000 | perfectly_correlated | T0 | 0.01183 | 0.9450 | 1.0000 | 0.1980 | 0.8095 | 1249.60 | 9998438 |
| positive_3pp | 5000 | perfectly_correlated | T1 | 0.01183 | 0.9435 | 1.0000 | 0.1965 | 0.8110 | 1249.61 | 9998517 |
| heterogeneous | 200 | conditionally_independent | U0 | 0.02772 | 0.9600 | 1.0000 | 0.0000 | 0.2675 | 50.13 | 560006 |
| heterogeneous | 200 | conditionally_independent | U1 | 0.02709 | 0.9635 | 1.0000 | 0.0000 | 0.3070 | 50.13 | 560006 |
| heterogeneous | 200 | conditionally_independent | ASI-adapted | 0.02688 | 0.9680 | 1.0000 | 0.0000 | 0.3155 | 31.50 | 560511 |
| heterogeneous | 200 | conditionally_independent | Robust-ASI-adapted | 0.02634 | 0.9625 | 1.0000 | 0.0000 | 0.3175 | 38.73 | 560246 |
| heterogeneous | 200 | conditionally_independent | T0 | 0.02670 | 0.9595 | 1.0000 | 0.0000 | 0.3155 | 44.84 | 559186 |
| heterogeneous | 200 | conditionally_independent | T1 | 0.02671 | 0.9605 | 1.0000 | 0.0000 | 0.3165 | 45.14 | 559332 |
| heterogeneous | 200 | perfectly_correlated | U0 | 0.05012 | 0.9420 | 1.0000 | 0.0000 | 0.1390 | 50.12 | 561100 |
| heterogeneous | 200 | perfectly_correlated | U1 | 0.05007 | 0.9435 | 1.0000 | 0.0000 | 0.1660 | 50.12 | 561100 |
| heterogeneous | 200 | perfectly_correlated | ASI-adapted | 0.04969 | 0.9395 | 1.0000 | 0.0000 | 0.1700 | 31.58 | 559418 |
| heterogeneous | 200 | perfectly_correlated | Robust-ASI-adapted | 0.04849 | 0.9505 | 1.0000 | 0.0000 | 0.1750 | 38.69 | 559423 |
| heterogeneous | 200 | perfectly_correlated | T0 | 0.04881 | 0.9440 | 1.0000 | 0.0000 | 0.1610 | 44.91 | 560290 |
| heterogeneous | 200 | perfectly_correlated | T1 | 0.04881 | 0.9440 | 1.0000 | 0.0000 | 0.1615 | 45.19 | 560288 |
| heterogeneous | 1000 | conditionally_independent | U0 | 0.01276 | 0.9525 | 1.0000 | 0.0075 | 0.7645 | 249.84 | 2792789 |
| heterogeneous | 1000 | conditionally_independent | U1 | 0.01251 | 0.9545 | 1.0000 | 0.0050 | 0.7720 | 249.84 | 2792789 |
| heterogeneous | 1000 | conditionally_independent | ASI-adapted | 0.01244 | 0.9540 | 1.0000 | 0.0000 | 0.7785 | 156.99 | 2792192 |
| heterogeneous | 1000 | conditionally_independent | Robust-ASI-adapted | 0.01208 | 0.9550 | 1.0000 | 0.0000 | 0.8030 | 193.10 | 2794181 |
| heterogeneous | 1000 | conditionally_independent | T0 | 0.01226 | 0.9535 | 1.0000 | 0.0000 | 0.7890 | 223.96 | 2793806 |
| heterogeneous | 1000 | conditionally_independent | T1 | 0.01228 | 0.9540 | 1.0000 | 0.0005 | 0.7860 | 225.39 | 2793244 |
| heterogeneous | 1000 | perfectly_correlated | U0 | 0.02196 | 0.9495 | 1.0000 | 0.0035 | 0.3900 | 249.85 | 2804702 |
| heterogeneous | 1000 | perfectly_correlated | U1 | 0.02176 | 0.9480 | 1.0000 | 0.0035 | 0.3995 | 249.85 | 2804702 |
| heterogeneous | 1000 | perfectly_correlated | ASI-adapted | 0.02159 | 0.9580 | 1.0000 | 0.0000 | 0.3920 | 157.41 | 2803870 |
| heterogeneous | 1000 | perfectly_correlated | Robust-ASI-adapted | 0.02075 | 0.9545 | 1.0000 | 0.0000 | 0.4045 | 193.16 | 2804534 |
| heterogeneous | 1000 | perfectly_correlated | T0 | 0.02133 | 0.9485 | 1.0000 | 0.0030 | 0.4075 | 223.97 | 2804565 |
| heterogeneous | 1000 | perfectly_correlated | T1 | 0.02130 | 0.9465 | 1.0000 | 0.0035 | 0.4055 | 225.47 | 2805414 |
| heterogeneous | 5000 | conditionally_independent | U0 | 0.00587 | 0.9450 | 1.0000 | 0.9575 | 1.0000 | 1248.93 | 13997536 |
| heterogeneous | 5000 | conditionally_independent | U1 | 0.00574 | 0.9475 | 1.0000 | 0.9540 | 1.0000 | 1248.93 | 13997536 |
| heterogeneous | 5000 | conditionally_independent | ASI-adapted | 0.00559 | 0.9505 | 1.0000 | 0.4510 | 1.0000 | 786.66 | 14005732 |
| heterogeneous | 5000 | conditionally_independent | Robust-ASI-adapted | 0.00549 | 0.9485 | 1.0000 | 0.9025 | 1.0000 | 965.34 | 14002230 |
| heterogeneous | 5000 | conditionally_independent | T0 | 0.00557 | 0.9505 | 1.0000 | 0.9510 | 1.0000 | 1119.28 | 13992345 |
| heterogeneous | 5000 | conditionally_independent | T1 | 0.00558 | 0.9500 | 1.0000 | 0.9505 | 1.0000 | 1126.56 | 13992136 |
| heterogeneous | 5000 | perfectly_correlated | U0 | 0.00982 | 0.9540 | 0.9995 | 0.4325 | 0.9145 | 1250.63 | 14010400 |
| heterogeneous | 5000 | perfectly_correlated | U1 | 0.00974 | 0.9525 | 0.9990 | 0.4265 | 0.9160 | 1250.63 | 14010400 |
| heterogeneous | 5000 | perfectly_correlated | ASI-adapted | 0.00969 | 0.9470 | 1.0000 | 0.1165 | 0.9185 | 787.62 | 14018915 |
| heterogeneous | 5000 | perfectly_correlated | Robust-ASI-adapted | 0.00952 | 0.9450 | 1.0000 | 0.3375 | 0.9315 | 966.79 | 14018964 |
| heterogeneous | 5000 | perfectly_correlated | T0 | 0.00959 | 0.9545 | 0.9995 | 0.4230 | 0.9240 | 1120.94 | 14014115 |
| heterogeneous | 5000 | perfectly_correlated | T1 | 0.00958 | 0.9530 | 0.9995 | 0.4215 | 0.9240 | 1128.30 | 14014616 |
| heterogeneous_wrong_uncertainty | 200 | conditionally_independent | U0 | 0.02772 | 0.9600 | 1.0000 | 0.0000 | 0.2675 | 50.13 | 560006 |
| heterogeneous_wrong_uncertainty | 200 | conditionally_independent | U1 | 0.02709 | 0.9635 | 1.0000 | 0.0000 | 0.3070 | 50.13 | 560006 |
| heterogeneous_wrong_uncertainty | 200 | conditionally_independent | ASI-adapted | 0.02716 | 0.9635 | 1.0000 | 0.0000 | 0.3095 | 50.13 | 557581 |
| heterogeneous_wrong_uncertainty | 200 | conditionally_independent | Robust-ASI-adapted | 0.03114 | 0.9645 | 1.0000 | 0.0000 | 0.2905 | 50.22 | 485094 |
| heterogeneous_wrong_uncertainty | 200 | conditionally_independent | T0 | 0.02670 | 0.9595 | 1.0000 | 0.0000 | 0.3155 | 44.84 | 559186 |
| heterogeneous_wrong_uncertainty | 200 | conditionally_independent | T1 | 0.04483 | 0.9750 | 1.0000 | 0.0000 | 0.2715 | 50.16 | 351850 |
| heterogeneous_wrong_uncertainty | 200 | perfectly_correlated | U0 | 0.05012 | 0.9420 | 1.0000 | 0.0000 | 0.1390 | 50.12 | 561100 |
| heterogeneous_wrong_uncertainty | 200 | perfectly_correlated | U1 | 0.05007 | 0.9435 | 1.0000 | 0.0000 | 0.1660 | 50.12 | 561100 |
| heterogeneous_wrong_uncertainty | 200 | perfectly_correlated | ASI-adapted | 0.05024 | 0.9425 | 1.0000 | 0.0000 | 0.1685 | 50.12 | 558751 |
| heterogeneous_wrong_uncertainty | 200 | perfectly_correlated | Robust-ASI-adapted | 0.05609 | 0.9450 | 1.0000 | 0.0000 | 0.1640 | 50.14 | 485432 |
| heterogeneous_wrong_uncertainty | 200 | perfectly_correlated | T0 | 0.04881 | 0.9440 | 1.0000 | 0.0000 | 0.1610 | 44.91 | 560290 |
| heterogeneous_wrong_uncertainty | 200 | perfectly_correlated | T1 | 0.07780 | 0.9310 | 1.0000 | 0.0000 | 0.1445 | 50.07 | 350487 |
| heterogeneous_wrong_uncertainty | 1000 | conditionally_independent | U0 | 0.01276 | 0.9525 | 1.0000 | 0.0075 | 0.7645 | 249.84 | 2792789 |
| heterogeneous_wrong_uncertainty | 1000 | conditionally_independent | U1 | 0.01251 | 0.9545 | 1.0000 | 0.0050 | 0.7720 | 249.84 | 2792789 |
| heterogeneous_wrong_uncertainty | 1000 | conditionally_independent | ASI-adapted | 0.01257 | 0.9535 | 1.0000 | 0.0050 | 0.7655 | 249.85 | 2782814 |
| heterogeneous_wrong_uncertainty | 1000 | conditionally_independent | Robust-ASI-adapted | 0.01414 | 0.9550 | 1.0000 | 0.0005 | 0.6985 | 249.97 | 2419393 |
| heterogeneous_wrong_uncertainty | 1000 | conditionally_independent | T0 | 0.01226 | 0.9535 | 1.0000 | 0.0000 | 0.7890 | 223.96 | 2793806 |
| heterogeneous_wrong_uncertainty | 1000 | conditionally_independent | T1 | 0.02044 | 0.9500 | 1.0000 | 0.0000 | 0.4920 | 250.29 | 1763303 |
| heterogeneous_wrong_uncertainty | 1000 | perfectly_correlated | U0 | 0.02196 | 0.9495 | 1.0000 | 0.0035 | 0.3900 | 249.85 | 2804702 |
| heterogeneous_wrong_uncertainty | 1000 | perfectly_correlated | U1 | 0.02176 | 0.9480 | 1.0000 | 0.0035 | 0.3995 | 249.85 | 2804702 |
| heterogeneous_wrong_uncertainty | 1000 | perfectly_correlated | ASI-adapted | 0.02186 | 0.9455 | 1.0000 | 0.0030 | 0.3995 | 249.88 | 2794392 |
| heterogeneous_wrong_uncertainty | 1000 | perfectly_correlated | Robust-ASI-adapted | 0.02477 | 0.9490 | 1.0000 | 0.0010 | 0.3470 | 249.77 | 2425947 |
| heterogeneous_wrong_uncertainty | 1000 | perfectly_correlated | T0 | 0.02133 | 0.9485 | 1.0000 | 0.0030 | 0.4075 | 223.97 | 2804565 |
| heterogeneous_wrong_uncertainty | 1000 | perfectly_correlated | T1 | 0.03525 | 0.9520 | 1.0000 | 0.0000 | 0.2445 | 249.60 | 1760066 |
| heterogeneous_wrong_uncertainty | 5000 | conditionally_independent | U0 | 0.00587 | 0.9450 | 1.0000 | 0.9575 | 1.0000 | 1248.93 | 13997536 |
| heterogeneous_wrong_uncertainty | 5000 | conditionally_independent | U1 | 0.00574 | 0.9475 | 1.0000 | 0.9540 | 1.0000 | 1248.93 | 13997536 |
| heterogeneous_wrong_uncertainty | 5000 | conditionally_independent | ASI-adapted | 0.00574 | 0.9460 | 1.0000 | 0.9555 | 1.0000 | 1248.97 | 13945008 |
| heterogeneous_wrong_uncertainty | 5000 | conditionally_independent | Robust-ASI-adapted | 0.00638 | 0.9470 | 1.0000 | 0.8280 | 0.9975 | 1249.51 | 12105022 |
| heterogeneous_wrong_uncertainty | 5000 | conditionally_independent | T0 | 0.00557 | 0.9505 | 1.0000 | 0.9510 | 1.0000 | 1119.28 | 13992345 |
| heterogeneous_wrong_uncertainty | 5000 | conditionally_independent | T1 | 0.00908 | 0.9450 | 1.0000 | 0.1145 | 0.9470 | 1249.10 | 8790303 |
| heterogeneous_wrong_uncertainty | 5000 | perfectly_correlated | U0 | 0.00982 | 0.9540 | 0.9995 | 0.4325 | 0.9145 | 1250.63 | 14010400 |
| heterogeneous_wrong_uncertainty | 5000 | perfectly_correlated | U1 | 0.00974 | 0.9525 | 0.9990 | 0.4265 | 0.9160 | 1250.63 | 14010400 |
| heterogeneous_wrong_uncertainty | 5000 | perfectly_correlated | ASI-adapted | 0.00976 | 0.9535 | 0.9990 | 0.4240 | 0.9155 | 1250.61 | 13958036 |
| heterogeneous_wrong_uncertainty | 5000 | perfectly_correlated | Robust-ASI-adapted | 0.01074 | 0.9570 | 1.0000 | 0.2570 | 0.8615 | 1250.41 | 12116562 |
| heterogeneous_wrong_uncertainty | 5000 | perfectly_correlated | T0 | 0.00959 | 0.9545 | 0.9995 | 0.4230 | 0.9240 | 1120.94 | 14014115 |
| heterogeneous_wrong_uncertainty | 5000 | perfectly_correlated | T1 | 0.01549 | 0.9525 | 1.0000 | 0.0305 | 0.5865 | 1251.04 | 8811251 |

## 统计结论与失败原因

有限样本保守区间和渐近Wald区间分开报告。保守界可能因全支持逆概率幅度而明显变宽，误差较小不自动对应更强检出能力。零收益和负收益环境的采用率为错误批准，正收益环境的采用率为检出能力。

总是拒绝采用的参照具有零错误批准和零检出能力。此参照没有被解释为评估方法成功。

每种采样设计在相同历史与权限下评估。主候选的风险是历史残差方向错误时重新分配预算会提高方差，公开鲁棒采样的收益也必须由实测比较决定。

## 80%检出率资源

| 环境 | 相关性 | 方法 | 推断 | 首个达到80%的题数 | 平均Gold | 平均Token |
|---|---|---|---|---|---|---|
| positive_1pp | conditionally_independent | U0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | U0 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | U1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | U1 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | ASI-adapted | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | Robust-ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | Robust-ASI-adapted | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | T0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | T0 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | conditionally_independent | T1 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | U0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | U0 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | U1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | U1 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | ASI-adapted | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | Robust-ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | Robust-ASI-adapted | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | T0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | T0 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_1pp | perfectly_correlated | T1 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | U0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | U0 | population_wald | 5000 | 1249.79 | 9998523 |
| positive_3pp | conditionally_independent | U1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | U1 | population_wald | 5000 | 1249.79 | 9998523 |
| positive_3pp | conditionally_independent | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | ASI-adapted | population_wald | 5000 | 1249.62 | 9997449 |
| positive_3pp | conditionally_independent | Robust-ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | Robust-ASI-adapted | population_wald | 5000 | 1249.79 | 9998974 |
| positive_3pp | conditionally_independent | T0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | T0 | population_wald | 5000 | 1249.56 | 9997025 |
| positive_3pp | conditionally_independent | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | conditionally_independent | T1 | population_wald | 5000 | 1249.51 | 9996597 |
| positive_3pp | perfectly_correlated | U0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | U0 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | U1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | U1 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | ASI-adapted | population_wald | 5000 | 1249.62 | 9998524 |
| positive_3pp | perfectly_correlated | Robust-ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | Robust-ASI-adapted | population_wald | 5000 | 1250.18 | 10002509 |
| positive_3pp | perfectly_correlated | T0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | T0 | population_wald | 5000 | 1249.60 | 9998438 |
| positive_3pp | perfectly_correlated | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| positive_3pp | perfectly_correlated | T1 | population_wald | 5000 | 1249.61 | 9998517 |
| heterogeneous | conditionally_independent | U0 | population_bernstein | 5000 | 1248.93 | 13997536 |
| heterogeneous | conditionally_independent | U0 | population_wald | 5000 | 1248.93 | 13997536 |
| heterogeneous | conditionally_independent | U1 | population_bernstein | 5000 | 1248.93 | 13997536 |
| heterogeneous | conditionally_independent | U1 | population_wald | 5000 | 1248.93 | 13997536 |
| heterogeneous | conditionally_independent | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | conditionally_independent | ASI-adapted | population_wald | 5000 | 786.66 | 14005732 |
| heterogeneous | conditionally_independent | Robust-ASI-adapted | population_bernstein | 5000 | 965.34 | 14002230 |
| heterogeneous | conditionally_independent | Robust-ASI-adapted | population_wald | 1000 | 193.10 | 2794181 |
| heterogeneous | conditionally_independent | T0 | population_bernstein | 5000 | 1119.28 | 13992345 |
| heterogeneous | conditionally_independent | T0 | population_wald | 5000 | 1119.28 | 13992345 |
| heterogeneous | conditionally_independent | T1 | population_bernstein | 5000 | 1126.56 | 13992136 |
| heterogeneous | conditionally_independent | T1 | population_wald | 5000 | 1126.56 | 13992136 |
| heterogeneous | perfectly_correlated | U0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | perfectly_correlated | U0 | population_wald | 5000 | 1250.63 | 14010400 |
| heterogeneous | perfectly_correlated | U1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | perfectly_correlated | U1 | population_wald | 5000 | 1250.63 | 14010400 |
| heterogeneous | perfectly_correlated | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | perfectly_correlated | ASI-adapted | population_wald | 5000 | 787.62 | 14018915 |
| heterogeneous | perfectly_correlated | Robust-ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | perfectly_correlated | Robust-ASI-adapted | population_wald | 5000 | 966.79 | 14018964 |
| heterogeneous | perfectly_correlated | T0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | perfectly_correlated | T0 | population_wald | 5000 | 1120.94 | 14014115 |
| heterogeneous | perfectly_correlated | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous | perfectly_correlated | T1 | population_wald | 5000 | 1128.30 | 14014616 |
| heterogeneous_wrong_uncertainty | conditionally_independent | U0 | population_bernstein | 5000 | 1248.93 | 13997536 |
| heterogeneous_wrong_uncertainty | conditionally_independent | U0 | population_wald | 5000 | 1248.93 | 13997536 |
| heterogeneous_wrong_uncertainty | conditionally_independent | U1 | population_bernstein | 5000 | 1248.93 | 13997536 |
| heterogeneous_wrong_uncertainty | conditionally_independent | U1 | population_wald | 5000 | 1248.93 | 13997536 |
| heterogeneous_wrong_uncertainty | conditionally_independent | ASI-adapted | population_bernstein | 5000 | 1248.97 | 13945008 |
| heterogeneous_wrong_uncertainty | conditionally_independent | ASI-adapted | population_wald | 5000 | 1248.97 | 13945008 |
| heterogeneous_wrong_uncertainty | conditionally_independent | Robust-ASI-adapted | population_bernstein | 5000 | 1249.51 | 12105022 |
| heterogeneous_wrong_uncertainty | conditionally_independent | Robust-ASI-adapted | population_wald | 5000 | 1249.51 | 12105022 |
| heterogeneous_wrong_uncertainty | conditionally_independent | T0 | population_bernstein | 5000 | 1119.28 | 13992345 |
| heterogeneous_wrong_uncertainty | conditionally_independent | T0 | population_wald | 5000 | 1119.28 | 13992345 |
| heterogeneous_wrong_uncertainty | conditionally_independent | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | conditionally_independent | T1 | population_wald | 5000 | 1249.10 | 8790303 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | U0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | U0 | population_wald | 5000 | 1250.63 | 14010400 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | U1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | U1 | population_wald | 5000 | 1250.63 | 14010400 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | ASI-adapted | population_wald | 5000 | 1250.61 | 13958036 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | Robust-ASI-adapted | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | Robust-ASI-adapted | population_wald | 5000 | 1250.41 | 12116562 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | T0 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | T0 | population_wald | 5000 | 1120.94 | 14014115 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | T1 | population_bernstein | 未达到，最大5000题 | 未确定 | 未确定 |
| heterogeneous_wrong_uncertainty | perfectly_correlated | T1 | population_wald | 未达到，最大5000题 | 未确定 | 未确定 |

## 下一步

这些结果验证合成总体中的条件性机制。真实有限题池的恢复精度与未来问题总体的覆盖分别解释，后续独立真实题评价需要另行冻结预算。
