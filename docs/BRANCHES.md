# 分支与归档索引

`main` 保存当前论文主线 E17–E22、完整冻结协议与脱敏结果，以及必要的共享框架和早期来源依赖。

其他研究方向、早期实验和阶段性探索保存在各自实验分支。分支更名只去掉 `codex/`，不移动对应提交，不改写任何历史实验文件。旧字符串可能仍存在于冻结报告或来源记录中，这是原始证据的一部分。

## 保留与恢复

- 旧 `main`：[`archive/main-before-paper-20261118`](https://github.com/fallenplum/MAB/tree/archive/main-before-paper-20261118)，提交 `bf1db55079fb38b3fd91311f4f3731e405db714a`。
- 整理前完整快照：[E22 源提交](https://github.com/fallenplum/MAB/tree/8f490a06ebe1153be2fcbeb42ad3944d63075b7d)。
- 本轮保留的原始字节：[frozen_manifest.json](../reproducibility/frozen_manifest.json)。
- 转为历史归档的原始文件版本：[archive_inventory.json](../reproducibility/archive_inventory.json)。
- 更名前后名称与固定提交：[branch_migration.json](../reproducibility/branch_migration.json)。

更名结果由远端分支列表与 commit SHA 逐一核验；迁移清单本身不代表执行完成。

## 名称映射

| 旧名称 | 目标名称 | 保持的提交 |
|---|---|---|
| `codex/a-fixed-k3-e2-e3` | `a-fixed-k3-e2-e3` | [`3750e75711a6`](https://github.com/fallenplum/MAB/commit/3750e75711a6877a2258020be3db8c52f4c7dd37) |
| `codex/b1-reliable-paired-evaluation` | `b1-reliable-paired-evaluation` | [`a8ca3eb66013`](https://github.com/fallenplum/MAB/commit/a8ca3eb660136de0cc0793d153e3561b8315283b) |
| `codex/ces-traceable-baseline-20261112` | `ces-traceable-baseline-20261112` | [`a701eabf0720`](https://github.com/fallenplum/MAB/commit/a701eabf0720b834d584c3f47ccd8e82fdcb4440) |
| `codex/conditional-gain-20261106` | `conditional-gain-20261106` | [`8e8093c3e1f3`](https://github.com/fallenplum/MAB/commit/8e8093c3e1f3748edd1417564a7665f025b53e78) |
| `codex/decision-acquisition-e3-20261107` | `decision-acquisition-e3-20261107` | [`45ff7ad04fa7`](https://github.com/fallenplum/MAB/commit/45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6) |
| `codex/e3-locked-mechanism-review-20261104` | `e3-locked-mechanism-review-20261104` | [`1fe57f540970`](https://github.com/fallenplum/MAB/commit/1fe57f540970ead0eac70b20cf1a29571fd30390) |
| `codex/e4-statistical-model-audit-20261107` | `e4-statistical-model-audit-20261107` | [`b5fa3317c77b`](https://github.com/fallenplum/MAB/commit/b5fa3317c77b1256037d734c93a001ecb19bfa28) |
| `codex/e5-paired-gain-validation` | `e5-paired-gain-validation` | [`7e06c632787f`](https://github.com/fallenplum/MAB/commit/7e06c632787f1de093732aed0250c8a30d3035b7) |
| `codex/e6-contrast-directed-feedback` | `e6-contrast-directed-feedback` | [`0c34a38bede9`](https://github.com/fallenplum/MAB/commit/0c34a38bede9fcf46359316058759b27fa9f5fea) |
| `codex/e7-opportunity-and-feedback-learning` | `e7-opportunity-and-feedback-learning` | [`a7bc3d9a7216`](https://github.com/fallenplum/MAB/commit/a7bc3d9a72160b3e6134cae76a1a13f47b8b992b) |
| `codex/e8-decision-focused-combination-learning` | `e8-decision-focused-combination-learning` | [`45767047d556`](https://github.com/fallenplum/MAB/commit/45767047d556e6cd9cac0e8905cf6dd81d5e3d4f) |
| `codex/e8-m4-mechanism-audit` | `e8-m4-mechanism-audit` | [`0dece5c291c5`](https://github.com/fallenplum/MAB/commit/0dece5c291c5ec644f690f892638f81543aed691) |
| `codex/e9-consistent-cost-sensitive-learning` | `e9-consistent-cost-sensitive-learning` | [`6502fd62932e`](https://github.com/fallenplum/MAB/commit/6502fd62932e91deb35e3b09449506afcf9f7e1c) |
| `codex/e10-input-prefill-matching-audit` | `e10-input-prefill-matching-audit` | [`d8431d35cb48`](https://github.com/fallenplum/MAB/commit/d8431d35cb48a1acb1bcb57d646c17ead0b60740) |
| `codex/e11-group-mean-shrinkage` | `e11-group-mean-shrinkage` | [`84af4c1bcc63`](https://github.com/fallenplum/MAB/commit/84af4c1bcc63389f9ef9224680292a0e33fb212c) |
| `codex/e12-shared-feedback-learning` | `e12-shared-feedback-learning` | [`46e475f2c6d3`](https://github.com/fallenplum/MAB/commit/46e475f2c6d394a20c8292c12ff1c3194a0be9b3) |
| `codex/e13-partial-combination-feedback` | `e13-partial-combination-feedback` | [`0114a7f744eb`](https://github.com/fallenplum/MAB/commit/0114a7f744eb39aff24e7e2eb2dbb82af66cdd82) |
| `codex/e14-controlled-mechanism-diagnostics` | `e14-controlled-mechanism-diagnostics` | [`9f84007bf4d5`](https://github.com/fallenplum/MAB/commit/9f84007bf4d5ebf516f20a8e0fc74b5e721c4526) |
| `codex/e15-feedback-structure-interaction` | `e15-feedback-structure-interaction` | [`39ba966be8fc`](https://github.com/fallenplum/MAB/commit/39ba966be8fce97867b23ecd4f0ae1a5f502e862) |
| `codex/e16-signal-replication-representation` | `e16-signal-replication-representation` | [`1f012ae7b9af`](https://github.com/fallenplum/MAB/commit/1f012ae7b9af54adc6a68fcf4417fab81a269982) |
| `codex/e17-supervision-distortion-audit` | `e17-supervision-distortion-audit` | [`94a8579f9c8d`](https://github.com/fallenplum/MAB/commit/94a8579f9c8dc16d1f6afb943cee6678c5a9792a) |
| `codex/e18-1-statistics-and-mechanisms` | `e18-1-statistics-and-mechanisms` | [`341a8a3c48ee`](https://github.com/fallenplum/MAB/commit/341a8a3c48ee95b7a38c1a5713ac659fccba8f6c) |
| `codex/e18-mechanism-and-confirmation-preparation` | `e18-mechanism-and-confirmation-preparation` | [`95c6c8a38a71`](https://github.com/fallenplum/MAB/commit/95c6c8a38a71866d6efdb224aec04b9a158adf17) |
| `codex/e19-symmetric-tie-constraints` | `e19-symmetric-tie-constraints` | [`07b01361d425`](https://github.com/fallenplum/MAB/commit/07b01361d425b089e49643f1ec5282eae0ac1ec0) |
| `codex/e20-independent-confirmation-preparation` | `e20-independent-confirmation-preparation` | [`031d32ee3491`](https://github.com/fallenplum/MAB/commit/031d32ee34911f58c535803c078584c424c8f6e3) |
| `codex/e20-p2-independent-confirmation` | `e20-p2-independent-confirmation` | [`21209351651a`](https://github.com/fallenplum/MAB/commit/21209351651a76abc782662c00ff4523703bbd8a) |
| `codex/e21-p3-external-mechanism` | `e21-p3-external-mechanism` | [`692e481ee344`](https://github.com/fallenplum/MAB/commit/692e481ee344d1dd7632c77633716882de599c55) |
| `codex/e22-mechanism-specificity` | `e22-mechanism-specificity` | [`8f490a06ebe1`](https://github.com/fallenplum/MAB/commit/8f490a06ebe1153be2fcbeb42ad3944d63075b7d) |
| `codex/fewshot-adaptation-20261107` | `fewshot-adaptation-20261107` | [`3a5b15020f06`](https://github.com/fallenplum/MAB/commit/3a5b15020f06eeeec15bc5c87a439dffb20be215) |
| `codex/fewshot-protocol-fix-20261107` | `fewshot-protocol-fix-20261107` | [`7a21e158fabe`](https://github.com/fallenplum/MAB/commit/7a21e158fabe56acbe3a40a3dd0e09c139c6c493) |
| `codex/m4-noise-aware-set-learning` | `m4-noise-aware-set-learning` | [`19ba4fd5316b`](https://github.com/fallenplum/MAB/commit/19ba4fd5316bd897036622dab6b02fcfd0e036fd) |

`experiment/e9-bayes-gate` 原名称没有 `codex/`，保持不变。

## 已有本地 clone 的更新

先处理本地未提交修改，然后只做快进更新：

```bash
git fetch origin --prune
git switch main
git pull --ff-only origin main
```

如本地 main 有自己的提交，`--ff-only` 会停止；先保留本地工作再处理分叉，不使用强制重置。

对于已更名的本地实验分支，按本表更新，例如：

```bash
git branch -m codex/e22-mechanism-specificity e22-mechanism-specificity
git branch --set-upstream-to=origin/e22-mechanism-specificity e22-mechanism-specificity
```

冻结源码中的 Windows 路径、历史分支名称与源文件哈希不做全局字符串替换。
