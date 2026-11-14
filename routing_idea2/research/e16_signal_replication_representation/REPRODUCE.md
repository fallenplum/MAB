# E16复现

原MAB Python3.11、NumPy、SciPy、scikit-learn和psutil。需要原train600输入与20组合×3生成输出，E13原H/E及PCA16状态，E8 M0_tuned各折FIT_LOG与model_state，E10 PREFILL_FEATURES_FROZEN.npz、原输入来源清单和编码冻结收据。

MAB_ASSET_ROOT指定原资产根，MAB_E16_ROOT可指定一个全新输出目录。复用E14 e14_common.py的不可覆盖JSON与哈希函数。

```powershell
python -B -X utf8 code/run_e16.py prepare
python -B -X utf8 code/run_e16.py freeze
python -B -X utf8 code/evaluate_e16.py
python -B -X utf8 code/finalize_e16.py
```

prepare执行所有源码AST与单元测试、Qwen行身份审计、原Ridge恢复，并封存内外层分折、嵌套样本顺序、seed子集、预算和模型选择规则。freeze仅使用各自合法H奖励，保存全部B/C模型与动作后再由evaluate读取E奖励及完整A诊断矩阵。

私有工件保留逐题预测、真实评分、采样题目身份、向量、权重及内折日志。公开文件为脱敏统计和源码。历史训练反馈和输入编码成本、当前缓存复用CPU成本分别记录。三个生成是题内重复，600独立题不扩大计数。

置信区间和bootstrap执行数为0。新语言模型生成和编码前向为0。validation200、fresh400和sealed200不读取。
