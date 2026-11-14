# E15复现

## 环境和私有资产

原MAB Python3.11、NumPy、SciPy、scikit-learn、psutil环境。需要E13完整private/worlds和inputs、原train600缓存及E3 assets；E14 public/code和private/prefixes用于逐项核对缓存复用；SaveRouter固定作者源码沿用原路径。

MAB_ASSET_ROOT设置实验资产根目录，MAB_E15_ROOT可指定全新输出目录，MAB_E14_CODE指定E14源码目录。私有权重、题目索引、反馈评分与原矩阵不包含于公开交付。

## 入口

```powershell
python -B -X utf8 code/run_e15.py prepare
python -B -X utf8 code/run_e15.py smoke
python -B -X utf8 code/run_e15.py freeze
python -B -X utf8 code/evaluate_e15.py
python -B -X utf8 code/finalize_e15.py
```

prepare先封存四个绝对预算、唯一主点、三项主要比较、停止条件、原输入来源和盲随机动作。smoke执行首条完整world资格，其模型被正式阶段复用。freeze在每种结构内为三学习器提供完全同一批反馈，封存全部模型和动作。评价器之后读取外折E奖励。

RandomPair从19候选均匀抽取，新增1/2/3成员并使用原聚合/成员成本缓存。各结构只把已选完整前缀的观测传给原学习器。完整计划池费用和被选前缀费用分开，物理调用为0。该设计属于回顾性固定缓存对比。

CommonCoverage与TokenMatched分开计费。前者匹配原题数量与身份，后者匹配Token上限。所有开发正负结果保留。原始题、答案、私有矩阵和权重留在private；public为脱敏表、协议和源码。置信区间、bootstrap和假设检验不执行，fresh400和sealed200不读取。
