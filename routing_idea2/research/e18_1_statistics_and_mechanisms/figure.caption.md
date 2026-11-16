# 联合证据图说明

**Figure. Controlled supervision interventions and frozen ensemble-routing evaluation.** The upper panel summarizes E17's semantic-preserving action renumberings on 600 development questions. Training targets, physical team identities and semantic inference priority were controlled separately. Physical-team agreement averaged 6.135% for SPO_coordinate and 100% for Ridge and TSPO_uniform. The inference-only control retained the effect of TSPO score near-ties under numeric priority. The middle panel fixes eight joint reward draws and retains the −0.889pp development difference of MeanThenLoss relative to LossThenMean. The lower panel reports the frozen E18 evaluation, with 2,998 scorable questions from 3,000 requested. The primary TSPO–SPO point difference was +0.01175pp, with 377 corrected and 376 damaged generation events. Its marginal 95% interval was [−0.66693,+0.69006]pp. This interval was added in E18.1 using 20,000 stratified paired question-cluster bootstrap resamples after the E18 results had been examined. All three generation repetitions stayed within the original question cluster.

图中百分比准确率采用GSM8K和BBH各50%的任务宏平均。E17的100次重编号共享600题，E18每题三个生成重复保持成簇。编号等变、代理总体目标和新题质量分别由对应证据支持。图保留了真实运算顺序对照的负结果、E18近零质量差及两道排除输入。

可编辑源为`figure.drawio`，同时提供`figure.svg`与`figure.png`。几何场景保存在`figure.geometry.json`。重现入口为`make_joint_evidence_figure.py`，随目录交付的`shared_scene_renderer.py`为既有`e18_mechanism_and_confirmation_preparation/make_mechanism_diagram.py`的原字节副本，旧文件保持原字节。该依赖SHA为`9bd5c45bd2f86d98c837fa844403e727b11a2073e0795ec3b3f22a3cbb3b2cf8`。

本机PATH与四个常见安装位置均未发现draw.io CLI。SVG由共享几何场景写出，PNG通过同一场景的Pillow渲染，单独的`.drawio`保留全部可编辑框、文字和连接线。原生CLI导出次数为0。结构检查为0错误、0警告、0交叉和0重叠，图像已进行两轮视觉核对。

来源统计为E18.1 `E18_1_COMPLETE_RESULTS.json`，SHA为`ab398af856c9daf14c93b4f35cb0eb407db522aad3c29457c0aea356bd8faefe`。本图生成没有调用模型、训练或bootstrap。
