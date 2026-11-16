# E17/E18/E19联合机制证据图

**Figure. From tied rewards to training constraints.** The upper row retains the earlier E17 champion and inference-priority interventions and E18 frozen quality result. The middle row separates a finite-repeat population construction from a shared-parameter construction with a symmetric penalty. The theoretical coefficient 60 belongs to the three-dimensional example. The lower row reports E19's completed endpoint comparison under fixed rewards, features, uniform empirical-champion weights, original 480-row normalization and fold-specific coefficient penalties. P0's two main and two reverse-mapping fits passed. Across P1's 600 original outer-fold questions, 150 had different strict winners at both endpoints. This region contributed +1.055556pp, with 38 corrected and 19 damaged recorded generation outcomes. Questions with a numerical tie contributed +0.386464pp. The prespecified numerical-argmax-uniform C−B effect was +1.442019pp, with a 10000-resample paired question-bootstrap 95% interval of [+0.190691,+2.722239] pp. The raw floating-point exact-argmax diagnostic was +1pp with an interval of [−0.333333,+2.333333] pp. These are conditional development results on the repeatedly used train600 pool.

图的三层分别保留历史端点现象、两个理论构造和本轮隔离对称约束强度的结果。150道严格胜者变化支持超出共同分数平移与推理并列的实际作用。全题收益还包含数值并列区贡献，图中单列两者。raw精确argmax诊断的区间跨零，数值服务定义的敏感性保留在主图。

原E17/E18/E18.1文件及图保持既有版本。当前图及渲染器仅在E19目录内生成。旧新B历史复核记录8道fold1问题的support13→14，整体B准确率影响−0.008242pp；fold3得分差但support不变。这些数值差异保留有限残差证书条件，本图的B/C均为本轮新拟合。

## 重现

在本目录的上一级E19代码目录执行

```powershell
python -B -X utf8 make_joint_mechanism_figure.py --summary FIGURE_PUBLIC_SUMMARY.json --output-root figures
```

`shared_scene_renderer.py`是从既有冻结渲染器选择复制的独立共享函数，旧文件保持原字节。draw.io原生CLI在当前环境缺失，`.drawio`保留逐个可编辑框、标签、连接线；SVG与PNG由相同几何场景本地输出。PNG为独立预览文件，几何保存于`figure.geometry.json`。新图不触发模型拟合、LLM调用、奖励读取或bootstrap。

## 数值来源

- 完整E19结果SHA为`b9d0d2d0c340c26572be7441a92c20ea68807c436d252e2c01198a1692812435`。
- 数值与访问复核SHA为`4fc39686625858059bb66a212c5b71ab44404cc7ae6740f3933c71bddf8f9533`。
- 历史数值恢复诊断SHA为`314cc81f2f72a57ee15475a2509ef03a3fac56d41bde099e3d3106fe5032a3d8`。
- 正式P0/P1协议SHA为`d9715da6b15fe5d35d34c4ec876cfd613284373b2a1d792674248c3b9348269c`。

图中E18.1的历史边际区间来自当时查看E18结果后开展的统计补充，E19则采用本轮预定10000次bootstrap。两项统计角色分别记录。100次重编号及同题三个生成重复均保留原始问题单位。
