# 实验分支

`main` 保存 E17–E24 论文主线的代码、协议和结果。[全部实验](../EXPERIMENTS.md)提供 E1–E24、V2 分析及补充实验的报告、结果、代码和运行说明。

33 个实验分支均有本阶段首页，以及 `reproducibility/README.md`、`docs/REPRODUCIBILITY.md`、`EXPERIMENTS.md`、`experiments/README.md` 和 `docs/BRANCHES.md`。在这些入口切换分支，可查看对应阶段的记录。

E24 的独立复验保留在 [`codex/e24-training-deployment-replication`](https://github.com/fallenplum/MAB/tree/codex/e24-training-deployment-replication)，固定结果为 [`d48fef3cb1e7`](https://github.com/fallenplum/MAB/commit/d48fef3cb1e77e876d4410be4933c5eb6cdb5738)。主线通过 [E24 论文结果](../paper/E24.md) 与实验记录相连。

## 实验版本

表中分别列出原实验提交和本次说明更新提交。原实验报告、代码、协议和结果的文件内容保持一致；原首页可以从原提交查看。

| 阶段 | 分支 | 原实验提交 | 说明更新提交 |
|---|---|---|---|
| E2/E3 | [`a-fixed-k3-e2-e3`](https://github.com/fallenplum/MAB/tree/a-fixed-k3-e2-e3) | [`3750e75711a6`](https://github.com/fallenplum/MAB/commit/3750e75711a6877a2258020be3db8c52f4c7dd37) | [`6fca5bb51a3e`](https://github.com/fallenplum/MAB/commit/6fca5bb51a3e8e5b96635a9b412a576961518cae) |
| E2 | [`e3-locked-mechanism-review-20261104`](https://github.com/fallenplum/MAB/tree/e3-locked-mechanism-review-20261104) | [`1fe57f540970`](https://github.com/fallenplum/MAB/commit/1fe57f540970ead0eac70b20cf1a29571fd30390) | [`60edef4a686e`](https://github.com/fallenplum/MAB/commit/60edef4a686e907216a0a114d4614bfe99a54869) |
| E3 | [`decision-acquisition-e3-20261107`](https://github.com/fallenplum/MAB/tree/decision-acquisition-e3-20261107) | [`45ff7ad04fa7`](https://github.com/fallenplum/MAB/commit/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6) | [`ed4f3a1c3b1c`](https://github.com/fallenplum/MAB/commit/ed4f3a1c3b1cc2bd1a4eb60358214701f54c65d9) |
| E4 | [`e4-statistical-model-audit-20261107`](https://github.com/fallenplum/MAB/tree/e4-statistical-model-audit-20261107) | [`b5fa3317c77b`](https://github.com/fallenplum/MAB/commit/b5fa3317c77b1256037d734c93a001ecb19bfa28) | [`154d65d1f712`](https://github.com/fallenplum/MAB/commit/154d65d1f7121f3b1871712793142c0082169f86) |
| E5 | [`e5-paired-gain-validation`](https://github.com/fallenplum/MAB/tree/e5-paired-gain-validation) | [`7e06c632787f`](https://github.com/fallenplum/MAB/commit/7e06c632787f1de093732aed0250c8a30d3035b7) | [`f152325aa0da`](https://github.com/fallenplum/MAB/commit/f152325aa0da8cb9f64abe7d146bd9d17a1f421b) |
| E6 | [`e6-contrast-directed-feedback`](https://github.com/fallenplum/MAB/tree/e6-contrast-directed-feedback) | [`0c34a38bede9`](https://github.com/fallenplum/MAB/commit/0c34a38bede9fcf46359316058759b27fa9f5fea) | [`6b155ff2bb99`](https://github.com/fallenplum/MAB/commit/6b155ff2bb998e39940ac480be7402588fe17a1f) |
| E7 | [`e7-opportunity-and-feedback-learning`](https://github.com/fallenplum/MAB/tree/e7-opportunity-and-feedback-learning) | [`a7bc3d9a7216`](https://github.com/fallenplum/MAB/commit/a7bc3d9a72160b3e6134cae76a1a13f47b8b992b) | [`60fb2072a4ba`](https://github.com/fallenplum/MAB/commit/60fb2072a4ba632f8bcd72840f484428923280e5) |
| E8 | [`e8-decision-focused-combination-learning`](https://github.com/fallenplum/MAB/tree/e8-decision-focused-combination-learning) | [`45767047d556`](https://github.com/fallenplum/MAB/commit/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f) | [`fde33fc0230f`](https://github.com/fallenplum/MAB/commit/fde33fc0230ffdbf70e8880c5f87a12389de7011) |
| E8 M4 检查 | [`e8-m4-mechanism-audit`](https://github.com/fallenplum/MAB/tree/e8-m4-mechanism-audit) | [`0dece5c291c5`](https://github.com/fallenplum/MAB/commit/0dece5c291c5ec644f690f892638f81543aed691) | [`fbd1cfa2cb7d`](https://github.com/fallenplum/MAB/commit/fbd1cfa2cb7da8eae93f010c4ffb97503136dc73) |
| E9 | [`e9-consistent-cost-sensitive-learning`](https://github.com/fallenplum/MAB/tree/e9-consistent-cost-sensitive-learning) | [`6502fd62932e`](https://github.com/fallenplum/MAB/commit/6502fd62932e91deb35e3b09449506afcf9f7e1c) | [`1e65571a2020`](https://github.com/fallenplum/MAB/commit/1e65571a2020c4410c8a5861aa0e3a9ba72e3688) |
| E9 Bayes 对照 | [`experiment/e9-bayes-gate`](https://github.com/fallenplum/MAB/tree/experiment/e9-bayes-gate) | [`fd68c774642b`](https://github.com/fallenplum/MAB/commit/fd68c774642bf3612fbfea09957fdb6fd555a05c) | [`ea51691b769c`](https://github.com/fallenplum/MAB/commit/ea51691b769c78744e2a41fe4628e7aebc846a10) |
| E10 | [`e10-input-prefill-matching-audit`](https://github.com/fallenplum/MAB/tree/e10-input-prefill-matching-audit) | [`d8431d35cb48`](https://github.com/fallenplum/MAB/commit/d8431d35cb48a1acb1bcb57d646c17ead0b60740) | [`32a266d2206b`](https://github.com/fallenplum/MAB/commit/32a266d2206b292a6b069a1f61422dde8ab87a85) |
| E11 | [`e11-group-mean-shrinkage`](https://github.com/fallenplum/MAB/tree/e11-group-mean-shrinkage) | [`84af4c1bcc63`](https://github.com/fallenplum/MAB/commit/84af4c1bcc63389f9ef9224680292a0e33fb212c) | [`64b97fe67d78`](https://github.com/fallenplum/MAB/commit/64b97fe67d78de3e05507e67f9f02739adc41c5c) |
| E12 | [`e12-shared-feedback-learning`](https://github.com/fallenplum/MAB/tree/e12-shared-feedback-learning) | [`46e475f2c6d3`](https://github.com/fallenplum/MAB/commit/46e475f2c6d394a20c8292c12ff1c3194a0be9b3) | [`5db8253a5e54`](https://github.com/fallenplum/MAB/commit/5db8253a5e5414dfda29989492ccdeb2ef0c007c) |
| E13 | [`e13-partial-combination-feedback`](https://github.com/fallenplum/MAB/tree/e13-partial-combination-feedback) | [`0114a7f744eb`](https://github.com/fallenplum/MAB/commit/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82) | [`26ac0f0a7457`](https://github.com/fallenplum/MAB/commit/26ac0f0a74570da67094bf19399d838d98ef6a3c) |
| E14 | [`e14-controlled-mechanism-diagnostics`](https://github.com/fallenplum/MAB/tree/e14-controlled-mechanism-diagnostics) | [`9f84007bf4d5`](https://github.com/fallenplum/MAB/commit/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526) | [`3ddea9b6d137`](https://github.com/fallenplum/MAB/commit/3ddea9b6d137e535709c44ebcf7abbd3e4526736) |
| E15 | [`e15-feedback-structure-interaction`](https://github.com/fallenplum/MAB/tree/e15-feedback-structure-interaction) | [`39ba966be8fc`](https://github.com/fallenplum/MAB/commit/39ba966be8fce97867b23ecd4f0ae1a5f502e862) | [`525a119a31f5`](https://github.com/fallenplum/MAB/commit/525a119a31f575296e32c781cae71d80964ae74e) |
| E16 | [`e16-signal-replication-representation`](https://github.com/fallenplum/MAB/tree/e16-signal-replication-representation) | [`1f012ae7b9af`](https://github.com/fallenplum/MAB/commit/1f012ae7b9af54adc6a68fcf4417fab81a269982) | [`d48149d7e5d3`](https://github.com/fallenplum/MAB/commit/d48149d7e5d393e05e0ee5fe6f68725dc40af011) |
| E17 | [`e17-supervision-distortion-audit`](https://github.com/fallenplum/MAB/tree/e17-supervision-distortion-audit) | [`94a8579f9c8d`](https://github.com/fallenplum/MAB/commit/94a8579f9c8dc16d1f6afb943cee6678c5a9792a) | [`bea5c6dc8dc1`](https://github.com/fallenplum/MAB/commit/bea5c6dc8dc15a1be7d1e5f9f3cf27c353073a87) |
| E18 | [`e18-mechanism-and-confirmation-preparation`](https://github.com/fallenplum/MAB/tree/e18-mechanism-and-confirmation-preparation) | [`95c6c8a38a71`](https://github.com/fallenplum/MAB/commit/95c6c8a38a71866d6efdb224aec04b9a158adf17) | [`b20cb62a2bae`](https://github.com/fallenplum/MAB/commit/b20cb62a2baee0d4e7e890eb0e85b7b07e20cf6e) |
| E18.1 | [`e18-1-statistics-and-mechanisms`](https://github.com/fallenplum/MAB/tree/e18-1-statistics-and-mechanisms) | [`341a8a3c48ee`](https://github.com/fallenplum/MAB/commit/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c) | [`6f3c3c11a370`](https://github.com/fallenplum/MAB/commit/6f3c3c11a3704de4f94f736d5ae4f23f5a2768a3) |
| E19 | [`e19-symmetric-tie-constraints`](https://github.com/fallenplum/MAB/tree/e19-symmetric-tie-constraints) | [`07b01361d425`](https://github.com/fallenplum/MAB/commit/07b01361d425b089e49643f1ec5282eae0ac1ec0) | [`ae164b167173`](https://github.com/fallenplum/MAB/commit/ae164b1671732952d499e78c75da9fea33eab556) |
| E20 准备 | [`e20-independent-confirmation-preparation`](https://github.com/fallenplum/MAB/tree/e20-independent-confirmation-preparation) | [`031d32ee3491`](https://github.com/fallenplum/MAB/commit/031d32ee34911f58c535803c078584c424c8f6e3) | [`562c683a900e`](https://github.com/fallenplum/MAB/commit/562c683a900ec600c382e00903450eb019330a62) |
| E20/P2 | [`e20-p2-independent-confirmation`](https://github.com/fallenplum/MAB/tree/e20-p2-independent-confirmation) | [`21209351651a`](https://github.com/fallenplum/MAB/commit/21209351651a76abc782662c00ff4523703bbd8a) | [`0aa3f367e3d1`](https://github.com/fallenplum/MAB/commit/0aa3f367e3d13ba755edb4d9d5387d0f8e827095) |
| E21/P3 | [`e21-p3-external-mechanism`](https://github.com/fallenplum/MAB/tree/e21-p3-external-mechanism) | [`692e481ee344`](https://github.com/fallenplum/MAB/commit/692e481ee344d1dd7632c77633716882de599c55) | [`65c0b604a9a7`](https://github.com/fallenplum/MAB/commit/65c0b604a9a7d0540088faf42c79cf389ee5379a) |
| E22 | [`e22-mechanism-specificity`](https://github.com/fallenplum/MAB/tree/e22-mechanism-specificity) | [`8f490a06ebe1`](https://github.com/fallenplum/MAB/commit/8f490a06ebe1153be2fcbeb42ad3944d63075b7d) | [`3b696b05198c`](https://github.com/fallenplum/MAB/commit/3b696b05198c6597c994c18b200ab5c2f28e6b4c) |
| E23 | [`e23-deployment-prior-and-references`](https://github.com/fallenplum/MAB/tree/e23-deployment-prior-and-references) | [`3958f9ceaac5`](https://github.com/fallenplum/MAB/commit/3958f9ceaac5b0b1de88bbd47d89ddd940833ff2) | [`2f2a793fea76`](https://github.com/fallenplum/MAB/commit/2f2a793fea769b9723543a898ce21679d712bc99) |
| B1 | [`b1-reliable-paired-evaluation`](https://github.com/fallenplum/MAB/tree/b1-reliable-paired-evaluation) | [`a8ca3eb66013`](https://github.com/fallenplum/MAB/commit/a8ca3eb660136de0cc0793d153e3561b8315283b) | [`218786e8a581`](https://github.com/fallenplum/MAB/commit/218786e8a5814ae709c2e54befe16fda6f85a1f9) |
| CES 对照 | [`ces-traceable-baseline-20261112`](https://github.com/fallenplum/MAB/tree/ces-traceable-baseline-20261112) | [`a701eabf0720`](https://github.com/fallenplum/MAB/commit/a701eabf0720b834d584c3f47ccd8e82fdcb4440) | [`867ca811c1c7`](https://github.com/fallenplum/MAB/commit/867ca811c1c7da580c4bd07d617b99e6306379e4) |
| 条件增益 | [`conditional-gain-20261106`](https://github.com/fallenplum/MAB/tree/conditional-gain-20261106) | [`8e8093c3e1f3`](https://github.com/fallenplum/MAB/commit/8e8093c3e1f3748edd1417564a7665f025b53e78) | [`3e5449ce1c65`](https://github.com/fallenplum/MAB/commit/3e5449ce1c65d00a9dd8a4e39375c180ec27753f) |
| 少样本适应 | [`fewshot-adaptation-20261107`](https://github.com/fallenplum/MAB/tree/fewshot-adaptation-20261107) | [`3a5b15020f06`](https://github.com/fallenplum/MAB/commit/3a5b15020f06eeeec15bc5c87a439dffb20be215) | [`cf030475514f`](https://github.com/fallenplum/MAB/commit/cf030475514fe2afa548b8149f22189dee533791) |
| 少样本协议修正 | [`fewshot-protocol-fix-20261107`](https://github.com/fallenplum/MAB/tree/fewshot-protocol-fix-20261107) | [`7a21e158fabe`](https://github.com/fallenplum/MAB/commit/7a21e158fabe56acbe3a40a3dd0e09c139c6c493) | [`8aaf110ad788`](https://github.com/fallenplum/MAB/commit/8aaf110ad788189bc01ed66fbdd6198466b901d0) |
| M4 噪声分析 | [`m4-noise-aware-set-learning`](https://github.com/fallenplum/MAB/tree/m4-noise-aware-set-learning) | [`19ba4fd5316b`](https://github.com/fallenplum/MAB/commit/19ba4fd5316bd897036622dab6b02fcfd0e036fd) | [`a1e9ca5c3657`](https://github.com/fallenplum/MAB/commit/a1e9ca5c36571334a8711afb7a7cbd1b6300cb2e) |

完整的文件校验与分支更新记录见 [navigation_update.json](../reproducibility/navigation_update.json)。

## 旧 main 备份

[archive/main-before-paper-20261118](https://github.com/fallenplum/MAB/tree/archive/main-before-paper-20261118) 保留整理前的 main，提交为 `bf1db55079fb38b3fd91311f4f3731e405db714a`。E1 汇报、E2 结果和 V2 分析的直接入口见 [E1 与旧 main 目录](../experiments/e01-initial-experiments/README.md)。

## 分支更名记录

当前仓库有 35 个分支：main、33 个实验分支和 1 个旧 main 备份。实验分支已去掉 `codex/` 前缀，名称映射见 [首次更名清单](../reproducibility/branch_migration.json) 和 [E23 更名清单](../reproducibility/branch_migration_e23.json)。

[首次整理记录](../reproducibility/reorganization_receipt.json)与 [E23 合并记录](../reproducibility/e23_integration_receipt.json)保留当时的提交和检查结果。后续实验分支沿用 `e24-topic` 的命名形式。

## 更新本地仓库

在已有本地仓库中获取分支更新：

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
```

已有本地旧名称的分支可按映射表改名，例如：

```bash
git branch -m codex/e22-mechanism-specificity e22-mechanism-specificity
git branch --set-upstream-to=origin/e22-mechanism-specificity e22-mechanism-specificity
```
