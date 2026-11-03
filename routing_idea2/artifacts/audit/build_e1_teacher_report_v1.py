"""Build a teacher-facing E1 report from verified artifacts, without modifying formal outputs."""
from pathlib import Path
import json,sys,hashlib,math,time,runpy
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R/'.tools/docx_runtime/site'),str(R/'src')]
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
A=R/'artifacts/audit/e1_failure_diagnosis_20261103_v2'
OUT=R/'release/e1_20261103'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(v) for v in f if v.strip()]
def font(run,size=10.5,bold=False,color=None):
 run.font.name='Calibri';run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
 run.font.size=Pt(size);run.bold=bold
 if color:run.font.color.rgb=RGBColor.from_string(color)
def text(doc,s,bold=False,size=10.5):
 p=doc.add_paragraph();p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.18
 font(p.add_run(s),size,bold);return p
def head(doc,s,level=1):
 p=doc.add_paragraph(style='Heading '+str(level));font(p.add_run(s),15 if level==1 else 11.5,True,'17365D')
 p.paragraph_format.space_after=Pt(8);p.paragraph_format.keep_with_next=True
 return p
def table(doc,headers,data,widths=None,size=9):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for i,h in enumerate(headers):t.rows[0].cells[i].text=str(h)
 for row in data:
  cells=t.add_row().cells
  for i,v in enumerate(row):cells[i].text=str(v)
 for ri,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
  if ri==0:
   trpr.append(OxmlElement('w:tblHeader'))
  for ci,cell in enumerate(row.cells):
   cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   if widths:cell.width=Cm(widths[ci])
   tcpr=cell._tc.get_or_add_tcPr()
   shade=OxmlElement('w:shd');shade.set(qn('w:val'),'clear');shade.set(qn('w:color'),'auto');shade.set(qn('w:fill'),'17365D' if ri==0 else ('F1F5F9' if ri%2 else 'FFFFFF'));tcpr.insert_element_before(shade,'w:noWrap','w:tcMar','w:textDirection','w:tcFitText','w:vAlign','w:hideMark')
   for p in cell.paragraphs:
    p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4);p.paragraph_format.line_spacing=1.08
    for run in p.runs:font(run,size,ri==0,'FFFFFF' if ri==0 else '1F2937')
 return t
def page(doc,n,title):
 if n>1:doc.add_page_break()
 text(doc,'E1 实验汇报  |  2026-11-03  |  '+str(n).zfill(2),size=9)
 head(doc,title)
def main():
 OUT.mkdir(parents=True,exist_ok=True);report=OUT/'E1_流程与实验结果汇报_20261103.docx';assert not report.exists()
 original=read(R/'results/e1/e1_diagnostics.json');audit=read(A/'audit_metrics.json');inp=read(A/'input_reconstruction.json');op=read(A/'option_text_diagnostic.json');physical=read(A/'physical_call_summary.json')
 assert original==read(R/'artifacts/private/server_migration_20261102/e1_post_control_independent_review_v1.json')
 assert physical['calls']==8800 and not audit['gsm_independent_strict_score_mismatches']
 manifest=read(R/'data/derived/split_manifest.json')
 evalids={x['question_id'] for x in manifest['records'] if x['split']=='e1_diagnostic_evaluation'}
 qs={q['question_id']:q for f in ['available_records.jsonl','gsm8k_questions.jsonl'] for q in rows(R/'data/derived'/f) if q['question_id'] in evalids}
 from budget_router.scoring import score_answer
 pool=read(R/'configs/pool_v1.json')['models'];membertable=[]
 for m in pool:
  rr=[r for r in rows(R/'results/raw/e1'/('members_'+m['alias'].lower()+'.jsonl')) if r['question_id'] in evalids]
  by={t:sum(score_answer(t,r['result']['text'],qs[r['question_id']]['reference'],subtask=qs[r['question_id']].get('subtask')).score for r in rr if r['task']==t)/100 for t in ['gsm8k','bbh']}
  membertable.append([m['alias'],f"{100*by['gsm8k']:.1f}%",f"{100*by['bbh']:.1f}%",f"{50*(by['gsm8k']+by['bbh']):.1f}%"])
 doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7)
 sec.top_margin=Cm(1.8);sec.bottom_margin=Cm(1.8);sec.left_margin=Cm(1.8);sec.right_margin=Cm(1.8)
 normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10.5);normal._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
 normal.paragraph_format.line_spacing=1.18;normal.paragraph_format.space_after=Pt(6)
 for name in ['Heading 1','Heading 2']:doc.styles[name].paragraph_format.keep_with_next=True
 header=sec.header.paragraphs[0];font(header.add_run('有限评测预算下的模型组合路由 · E1 开发诊断'),8.5,color='64748B')
 footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
 font(footer.add_run('研究过程汇报  |  第 '),8,color='64748B')
 field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field);font(footer.add_run(' 页'),8,color='64748B')
 doc.core_properties.title='E1 流程与实验结果汇报'
 doc.core_properties.subject='真实8800次调用、聚合器退化与GSM8K数据边界审计'
 doc.core_properties.author='研究项目组'
 doc.core_properties.keywords='E1; GSM8K; BBH; 组合路由; 聚合器; 负结果; 只读审计'
 page(doc,1,'当前结论：尚未证明组合路由的实际收益')
 text(doc,'本报告覆盖已完成的原 E1 实验及其后验问题审计。原 E1 共完成 8800 次真实模型调用，独立重放与原结果一致；新增 600 次聚合器因果诊断单独记账，报告编制快照（2026-11-03 19:15）时尚未完成；追加完整结果见第11页。',bold=True)
 table(doc,['项目','已核实事实'],[
 ['实验规模','6 个冻结成员选 3 个，20 种组合；开发题 300 道，其中选择集 100、诊断评估集 200。'],
 ['生成重复','原 E1 只有 1 份生成缓存，主生成种子 42；bootstrap 不等于重新生成。'],
 ['强单模型','Qwen3-8B：GSM8K 95.0%、BBH 64.0%、等权宏平均 79.5%。'],
 ['校准所选固定组合','GSM8K 87.0%、BBH 52.0%、宏平均 69.5%；每题总 token 约为强单模型的 5.31 倍。'],
 ['关键复核','同批强模型三采样：直接投票 82.5%，原聚合器融合 76.5%。'],
 ['数据边界','E1 的 GSM8K 评估题来自官方 train；不是官方完整 test 成绩。'],
 ],[3.1,14.1],9.5)
 head(doc,'三项核心判断',2)
 text(doc,'1. 现有配置出现“质量下降、token 开销增加”的负结果，不能把 E1 的 PASS 写成方法有效。表中的固定组合也不是已经学到的动态路由策略。')
 text(doc,'2. 当前聚合环节存在可直接观测的损失；原因至少涉及输出截断、格式遵循与判断错误，需要用同候选、同问题对照分离。')
 text(doc,'3. 未发现标准答案被程序写入模型请求的证据；但官方训练题可能已进入模型预训练或后训练，现有审计不能排除这种污染。')
 text(doc,'置信度：上述缓存成绩、调用数及消息复核为高；“换强聚合器可恢复收益”为尚待检验；“模型记忆导致95%”目前证据不足。',size=9.5)
 page(doc,2,'研究问题与 E1 的完整流程')
 text(doc,'研究目标是在有限反馈预算下学习按题选择固定大小的模型组合。E1 是进入在线学习前的机会诊断：先检查候选互补性能否转化为冻结聚合器的最终正确答案，再考虑学习能否找出合适组合。')
 table(doc,['步骤','输入与操作','产物 / 检验'],[
 ['01','公开数据来源与目标对齐；固定问题划分','题目/参考仅用于离线准备与评分；生成请求排除金标。'],
 ['02','固定 6 个模型、版本、NF4、提示与参数','E0 协议锁；资格输出不算 E1 成绩。'],
 ['03','300 道 E1 题，各调用 6 个成员','1800 条成员回答；后续组合复用，避免重复计费。'],
 ['04','每题枚举 20 个无序三模型组合','6000 次真实聚合；候选顺序按题和种子确定。'],
 ['05','在选择集100题选强单模型、固定组合、独立Top-K','只用选择集标签确定这些基线；不拿评估集择优冒充可部署策略。'],
 ['06','评估集200题补足必要强对照','聚合器独答200；强模型新采样600；三采样聚合200。'],
 ['07','冻结评分器逐条评分，分任务与宏平均汇总','配对题级 bootstrap；实际调用费用与逻辑服务费用分开。'],
 ['08','只读审计与门槛解释','检查聚合损失、采样投票、消息泄漏、数据来源与成本。'],
 ],[1.1,7.7,8.4],9.3)
 head(doc,'PASS 具体表示什么',2)
 text(doc,'原门槛是工程设定：逐题理想组合上界相对最强可用基线，观测提升至少1个百分点且配对区间下界大于0。当前上界92.5%，相对强单模型79.5%高13个百分点，原95%区间为 +9 至 +18 个百分点。')
 text(doc,'理想上界是在看到本题真实标签后，从20个聚合结果中取最好值。它使用了不可部署的信息，只说明这份缓存存在机会，不证明路由可学、组合成本划算或全研究完成。',bold=True)
 text(doc,'后续 12000 次在线聚合目前因本次问题审计暂缓。已完成资产保留；未提前评分最终封存集。',size=9.5)
 page(doc,3,'数据划分、模型池与冻结参数')
 table(doc,['划分','GSM8K','BBH','用途'],[
 ['E1 选择集','50（官方train）','50（官方test）','选基线；不拟合后续PCA/在线路由。'],
 ['E1 诊断评估集','100（官方train）','100（官方test）','本报告主要结果与后验问题审计。'],
 ['在线校准集','100（官方train）','100（官方test）','后续200题，单独于E1。'],
 ['在线流','300（官方train）','300（官方test）','后续600题，预算反馈实验。'],
 ['最终封存集','100（官方test）','100（官方test）','满足后续释放条件后才做最终评估。'],
 ['资格题','4','4','技术资格；不进入主要评估准确率。'],
 ],[3.2,4.2,4.2,5.6],9.1)
 text(doc,'项目内部划分按题固定并防重叠，BBH在子任务间平衡；BBH资格来源是六模型历史缓存的完整可用交集，正式回答重新生成。缓存旧分数仅作来源字段与有效性检查，没有按高准确率筛题。',size=9.5)
 head(doc,'六个成员在 E1 评估集的单独成绩',2)
 table(doc,['模型','GSM8K','BBH','宏平均'],membertable,[10.0,2.4,2.4,2.4],8.5)
 text(doc,'成员：temperature=0.7、top_p=0.95、输出上限2048、输入上限4096；Qwen3显式关闭thinking。聚合器：Qwen2.5-7B-Instruct、temperature=0、top_p=1、输出上限256、输入上限6144。均按固定版本冻结，NF4，FP16计算。',size=9.2)
 text(doc,'每个候选按聚合器 tokenizer 保留最多512个源token（头128、尾384），省略标记另计实际输入；原成员完整生成费用始终计入。',size=9.2)
 page(doc,4,'实际执行过程、账本与硬件边界')
 table(doc,['调用组','问题与次数','真实调用数','prompt + completion token'],[
 ['普通成员','300题 × 6模型','1800',f"{sum(x['prompt_tokens']+x['completion_tokens'] for x in physical['groups'] if x['file'].startswith('members_')):,}"],
 ['三模型组合聚合','300题 × 20组合','6000','4,986,862'],
 ['聚合器独答','评估集200题','200','79,380'],
 ['强模型独立新采样','200题 × 3次','600','186,339'],
 ['强模型三采样聚合','评估集200题','200','142,969'],
 ['合计','不含E0资格与新增诊断','8800','6,075,463'],
 ],[4.2,5.3,2.0,5.7],9.2)
 head(doc,'可核验的执行时间',2)
 text(doc,'原成员首条调用始于北京时间2026-10-31 00:50；6000次组合聚合从11-01 09:59持续到11-02 20:55附近；聚合器独答于11-02晚完成；强模型600次新采样于11-03 08:33附近完成；最后200次三采样聚合于11-03 09:16附近完成。末时刻由记录的开始时间加调用耗时估算，不等同于批次结算时间。')
 text(doc,'原记录的模型调用耗时相加为250140秒，约69.48小时。它包含调用期间的等待与设备共享等因素，不是独占GPU计算时长，也不能与多卡日历用时直接比较。')
 head(doc,'恢复与迁移没有替代科学验收',2)
 text(doc,'主要E1输出在本机RTX 4060 Laptop 8GB上生成，末200次强三采样聚合在RTX A6000上完成。生成记录、实际token与原种子保留；跨硬件不能默认逐位数值等价。')
 text(doc,'Windows加载与长解码问题通过保留缓存、在成功落盘后的安全边界恢复处理。进程启动、权重就绪、技术资格及软件测试均没有被当作方法性能结果。')
 text(doc,'独立复核确认：原8800次成功记录完整，原E1汇总与独立重放一致；0条失败记录混入评分、0条费用不明记录。镜像和迁移副本没有重复计入真实调用数。',bold=True)
 page(doc,5,'主要结果：质量与 token 成本')
 metricnames=[('最强单模型（选择集选出）','strongest_single_calibration'),('固定三模型组合（选择集选出）','best_fixed_calibration'),('按单模型排名的独立Top-3','independent_topk_calibration'),('聚合器独答','aggregator_alone'),('强模型三采样再聚合','strong_model_resample'),('逐题理想组合上界（不可部署）','diagnostic_oracle_upper_reference')]
 data=[]
 for label,key in metricnames:
  v=original['metrics'][key];costkey='oracle_selected_diagnostic_only' if key=='diagnostic_oracle_upper_reference' else key
  data.append([label,f"{100*v['per_task']['gsm8k']:.1f}%",f"{100*v['per_task']['bbh']:.1f}%",f"{100*v['mean']:.1f}%",f"{original['costs'][costkey]:,.2f}"])
 data.append(['强模型三采样直接投票（后验诊断）','93.0%','72.0%','82.5%','931.70'])
 table(doc,['方法','GSM8K','BBH','宏平均','总token/题'],data,[7.2,2.1,2.1,2.1,3.7],8.8)
 text(doc,'统计口径：诊断评估200题，各任务100题；宏平均给GSM8K和BBH相同权重。每题总token包含全部成员调用和聚合调用的prompt与completion；不是美元成本或等时延预算。',size=9.3)
 head(doc,'如何解释这些结果',2)
 text(doc,'固定组合为Qwen3-8B、Qwen2.5-Coder-7B-Instruct、DeepHermes-3-Llama-3-8B-Preview。与强单模型相比，宏平均低10个百分点，token开销约5.31倍。独立Top-3为76.0%，同样没有超过强单模型。')
 text(doc,'事后看全部20个固定组合，评估集最高78.0%（Qwen3、DeepSeek、Cogito）；这是后验诊断数值，不能替代选择集选择结果，也不能叫已学会的组合路由。')
 text(doc,'三采样投票比原三采样聚合高6.0个百分点，题级分任务配对95%区间 [+3.0,+9.5]；比强单模型高3.0个百分点，区间 [-0.5,+7.0]。后者尚不足以证明稳定提升。')
 text(doc,'三采样再聚合比单次低3.0个百分点，但配对区间 [-8.0,+1.5] 仍包含零。上述新增区间基于5000次题级重采样，条件于已有生成，不代表多生成种子稳健性。')
 text(doc,'不能把20种组合的独立服务费用直接相加，当作本次诊断实际费用；诊断中成员回答被复用。原8800次实际token为6,075,463。',size=9.3)
 page(doc,6,'失败归因：正确候选在哪里丢失')
 table(doc,['固定组合的200题','计数','含义'],[
 ['至少一个成员答对','170','候选覆盖率85.0%，不等于最终准确率。'],
 ['最终聚合答对','139','最终准确率69.5%。'],
 ['有正确候选但聚合答错','33','其中11题触及输出上限、22题未触及。'],
 ['三个候选都错但聚合答对','2','聚合器也存在独立纠错，不能只计损失。'],
 ['三个候选都对但聚合答错','0 / 61','此项针对校准选出的固定组合。'],
 ['格式无法解析','22','其中BBH18题；不能全部归为知识错误。'],
 ],[6.3,2.0,8.9],9.3)
 text(doc,'净损失为33−2=31题，即85.0%覆盖率与69.5%准确率之间的15.5个百分点。33题具体损失不应误写成31题。',size=9.5)
 head(doc,'强模型三采样：同一批候选，融合方式不同',2)
 text(doc,'600次新采样按每题sample_index 0、1、2分组；直接投票取已解析答案多数，平票按较小sample_index，整个规则不使用真实标签。三个单次样本宏平均分别79.5%、80.0%、81.5%；投票82.5%，融合76.5%。')
 table(doc,['三采样全部答对、融合评分错的4题','归因'],[
 ['2题','答案内容正确，但输出选项文字或数值，没有按要求给选项字母。'],
 ['1题','推理已经得到正确状态，但在256 token处截断，缺最终答案行。'],
 ['1题','确实改成错误选项，属于聚合判断错误。'],
 ],[9.1,8.1],9.1)
 text(doc,'单独将最终答案与原题选项文字做唯一精确匹配，三采样融合可额外识别5个正确答案，76.5%变为79.0%；仍低于直接投票82.5%。这只是后验格式诊断，没有改写原评分。')
 text(doc,'候选压缩复核：7942次正确候选展示的最终答案全部在压缩后保留。它排除了这批数据中“裁掉最终正确答案”这一解释，但不能排除省略中间推理影响融合。',size=9.4)
 page(doc,7,'GSM8K 的95%：评分、泄漏与污染边界')
 text(doc,'95%是这100道官方train题中95道答对，不能当成官方test全集成绩，也不能仅凭数值证明泄漏或证明模型正常泛化。',bold=True)
 table(doc,['审计项','实际核验结果'],[
 ['官方题与答案对齐','E1 GSM8K 150题（含选择集）原题逐字匹配；最终数值参考150/150一致。BBH 150题官方题与target一致。'],
 ['成员生成请求','1800条普通成员与600条新采样请求，全部等于固定独答提示加官方原题。'],
 ['聚合输入内容','6000条普通聚合、200条强三采样聚合、200条独答请求，均从原题与实际候选精确重建一致。'],
 ['独立数值复评分','评估集100道强单模型GSM8K输出逐题独立解析复核，仍为95/100；百分号等显式单位按题目答案约定处理。'],
 ['评分器行为','比较最终答案，不因推理正文出现过正确数字就计对；不执行模型生成的算式代码。'],
 ['数据划分与筛题','项目内部按题防重叠、固定种子与子任务平衡；未按历史高分筛选问题。'],
 ],[3.6,13.6],9.3)
 head(doc,'已排除什么，仍不知道什么',2)
 text(doc,'已查范围内，没有发现程序把标准参考答案附加给模型，也没有发现数值评分把95%虚增。官方GSM8K train/test原题文本没有精确重叠。')
 text(doc,'仍无法排除：Qwen3等公开模型在预训练或后训练中见过GSM8K训练题及其解答；仅靠项目内题级不重叠，不能解决模型既有污染。该95%也只有100题，Wilson 95%区间约[88.8%,97.8%]，不是模型总体能力的精确常数。')
 text(doc,'GSM8K与BBH不能只按两个准确率横向判断泄漏：它们任务族、答案类型及难度结构不同，BBH还包含逻辑追踪、排序等子任务；本次强单模型BBH有7题格式无法解析。')
 text(doc,'关于污染的文献支持“需要独立新题复核”，并不能证明本项目的Qwen3已污染。GSM1K论文[3]发现部分模型在匹配风格的新题上下降，也发现部分模型仍有较好泛化。')
 page(doc,8,'后续安排：先完成隔离的聚合器因果诊断')
 text(doc,'保持原E0、原E1结果与封存题边界。新增诊断是后验开发实验，全部单列，不混入原8800次，也不把其结果当作新鲜测试。本页记录19:15运行快照；完整新增结果见第11页。',bold=True)
 table(doc,['组别','模型 / 输出上限','候选输入','新调用'],[
 ['现有对照','Qwen2.5-7B / 256','已有同批强模型三采样','0（复用已有200次）'],
 ['A：放宽预算','Qwen2.5-7B / 2048','逐字相同的既有聚合消息','200'],
 ['B：更换聚合器','Qwen3-8B / 2048','逐字相同的既有聚合消息','200'],
 ['C：强聚合器独答','Qwen3-8B / 2048','原独答布局，无候选','200'],
 ],[3.5,5.2,6.0,2.5],9.2)
 text(doc,'三组temperature=0、top_p=1、seed=42、NF4、FP16，Qwen3关闭thinking。B沿用原Qwen2.5 tokenizer得到的候选字符串，不重新裁剪；模型各自固定chat template仍有差别。')
 head(doc,'每个对照回答什么问题',2)
 text(doc,'A对现有对照：输出上限是否限制结果。B对A：在同输出预算和同消息下，更强聚合模型能否减少损失。B对C：收益是否确实来自候选，而非仅来自更强模型、不同提示或更多计算。')
 head(doc,'优势与风险',2)
 text(doc,'优势：复用已有候选、逐题配对，新增规模固定为600次；每个因素有明确对照，能避免直接重跑全套12000次而仍无法解释原因。')
 text(doc,'风险：仍使用E1开发题，不能证明新题泛化；提高上限可能增加成本；Qwen3并非必然更适合聚合；即使质量提高，仍须与强单模型、三采样投票及预算匹配基线比较。')
 text(doc,'决策条件：若强聚合没有超过同设置独答，或收益仍不足以支持成本，则保留负结果并重新评估研究设定。只有形成明确改动依据，才另行冻结新协议；不把旧版结果与新版结果拼成同一实验。')
 text(doc,'后续E2–E5的方法比较、延迟/预算消融及最终封存评测尚未完成。现有成员矩阵4800/4800、校准聚合4000/4000已在服务器完成；在线12000次聚合继续暂缓。',size=9.4)
 page(doc,9,'重要文件索引：老师可从哪里核查')
 text(doc,'GitHub交付以 routing_idea2/ 为实验源码根；报告放在 reports/e1/。逐题问题、参考答案、原始输出和权重留在本地，汇总结果与来源哈希进入仓库。',size=9.6)
 table(doc,['文件 / 目录','作用'],[
 ['reports/e1/本Word文档','面向汇报：流程、原E1结果、问题审计及后续边界。'],
 ['results/e1/e1_diagnostics.json','原E1汇总结果；包含均值、区间、成本与原机会门解释。'],
 ['results/e1/audit_metrics.json','后验复核：投票、聚合损失、输出上限、后验固定组合比较。'],
 ['results/e1/causal_diagnostic_summary.json','新增600次完整对照及独立核验；不混入原E1。'],
 ['results/e1/input_reconstruction.json','消息内容精确重建与正确候选最终答案保留检查。'],
 ['results/e1/option_text_diagnostic.json','只保留格式诊断汇总，原评分不变。'],
 ['results/e1/physical_call_summary.json','原8800次调用分组、token、调用耗时与原文件哈希。'],
 ['scripts/evaluate_real_e1.py','读取完整真实输出并组装E1；缺结果时拒绝冒充完成。'],
 ['src/budget_router/diagnostics.py','E1统计、固定组合选择、覆盖率与理想上界门。'],
 ['src/budget_router/e1_controls.py','选择集选强模型、构建三次独立新采样与融合对照。'],
 ['src/budget_router/scoring.py','GSM8K/BBH固定、安全的最终答案评分。'],
 ['src/budget_router/execution.py','匿名聚合提示、候选裁剪、内容哈希与执行账本。'],
 ['src/budget_router/member_execution.py','成员独答提示、任务验证与缓存复用。'],
 ['configs/；protocol/E0_LOCK.json','固定模型版本、参数和原E0哈希约束。'],
 ['artifacts/audit/audit_e1_*.py','本次只读审计脚本，不生成新模型答案。'],
 ],[7.1,10.1],8.8)
 text(doc,'完整源码中附带的后续阶段模块仅用于依赖与工程归档，不表示E2–E5已完成。新增600次诊断的服务器私有路径、进程和认证信息不进入报告或仓库。',size=9.2)
 page(doc,10,'复核说明、解释边界与参考文献')
 head(doc,'如何复核原E1',2)
 text(doc,'1. 核对源码、配置、E0锁及清单哈希；恢复本地私有问题、参考答案、逐题输出与固定权重来源。仅有GitHub汇总包，不能独立从原始输出复评分。')
 text(doc,'2. 使用原环境运行 scripts/evaluate_real_e1.py，向一个全新的输出路径写入独立审查结果，再与原 results/e1/e1_diagnostics.json 做完整字典比较。原输出路径拒绝覆盖。')
 text(doc,'示例：python scripts/evaluate_real_e1.py --root <实验根目录> --output <全新复核输出.json>',size=9.2)
 text(doc,'3. 本次后验审计脚本位于artifacts/audit/；输入必须匹配清单。审计采用独立数值复核与逐条消息内容重建；原E1正式评分结果保持不变。')
 head(doc,'汇报时避免的三种误读',2)
 text(doc,'“E1 PASS”等于组合路由有效：不成立，原门仅检验不可部署的理想上界机会。')
 text(doc,'“三采样76.5%”等于多数投票失败：不成立，该数是三个新样本再经弱聚合器的最终成绩。')
 text(doc,'“GSM8K95%”等于官方测试泛化或必然泄漏：都不成立，实际为100道官方训练题成绩；程序泄漏审计与模型既有污染是两个问题。')
 head(doc,'参考文献与可核查来源',2)
 for s in [
 '[1] Wang et al. Mixture-of-Agents Enhances Large Language Model Capabilities. 2024. https://arxiv.org/abs/2406.04692 。多层MoA的结果主要来自AlpacaEval 2.0、MT-Bench和FLASK，不能保证当前算术/逻辑配置有效。',
 '[2] Li et al. Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial? 2025. https://arxiv.org/abs/2502.00674 。强调候选质量与多样性的权衡，Self-MoA在多种场景超过异构MoA。',
 '[3] Zhang et al. A Careful Examination of Large Language Model Performance on Grade School Arithmetic. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2405.00332 。用匹配风格的新题研究污染与泛化，不能据此断言本项目特定模型污染。',
 '[4] 本项目原始汇总、独立重放、消息重建及费用账本。完整SHA256见交付清单；原E1正式结果文件未被后验诊断覆盖。'
 ]:text(doc,s,size=9.2)
 text(doc,'报告结论：当前配置没有证明组合路由的实际收益；聚合器及其输出接口存在可复现损失；95%的程序评分已复核，但数据污染风险仍待独立新题验证。',bold=True,size=10)
 causal_path=A/'causal_diagnostic_600_v1/report_summary.json'
 if causal_path.exists():
  c=read(causal_path)
  page(doc,11,'补充：600 次聚合器诊断已完成')
  text(doc,'本页为原E1之后追加的后验开发诊断。三组各200次真实调用均完成请求、响应与落盘校验，0条费用不明；与原8800次分开记账，合计9400次，不含资格和后续主实验。',bold=True)
  data=[['原Qwen2.5 / 256（已有）','93.0%','60.0%','76.5%','1,646.54']]
  for key,label in [('qwen25_long','Qwen2.5 / 2048 融合'),('qwen3_long_fuse','Qwen3 / 2048 融合'),('qwen3_long_alone','Qwen3 / 2048 独答')]:
   v=c['conditions'][key];data.append([label,f"{100*v['per_task']['gsm8k']['accuracy']:.1f}%",f"{100*v['per_task']['bbh']['accuracy']:.1f}%",f"{100*v['accuracy']:.1f}%",f"{v['logical_total_tokens_per_question']:,.2f}"])
  data.append(['原三采样直接投票','93.0%','72.0%','82.5%','931.70'])
  table(doc,['对照','GSM8K','BBH','宏平均','总token/题'],data,[7.2,2.1,2.1,2.1,3.7],9)
  text(doc,'融合成本含三次成员采样与一次聚合，独答和投票按各自完整服务计费；不是美元或归一化GPU时间。新增600次实际消耗366,092 token，原E1与新增诊断合计6,441,555 token。',size=9.4)
  head(doc,'哪些原因获得了更直接的证据',2)
  text(doc,'输出上限确有贡献：193个原未截断输出逐字复现，7个原截断输出全部是新输出前缀；延长后额外答对4道BBH题，宏平均提高2个百分点，配对95%区间[+0.5,+4.0]。')
  text(doc,'聚合模型也有影响：同消息、同2048上限时，Qwen3为83.0%，Qwen2.5为78.5%，差4.5个百分点，探索性配对区间[+1.0,+8.0]。模型各自chat template不同，不能把差异只归给权重规模。')
  head(doc,'为什么仍不能宣布组合路由有效',2)
  text(doc,'Qwen3融合比同设置独答高3.5个百分点，区间[-0.5,+7.5]；比无需新模型调用的直接投票仅高0.5个百分点，区间[-2.0,+3.0]。两项都仍包含零，融合的完整token开销约为投票的1.81倍。')
  text(doc,'本轮候选是同一个Qwen3的三次采样，尚未验证异构三模型组合的收益；题目仍是同一批E1开发题。上述区间条件于已有输出，属于后验探索，不是独立新测试或多生成种子结论。',size=9.6)
  text(doc,'当前决策：保留原V1负结果及本次修复证据。聚合器和输出上限解释了部分退化，但简单投票仍是必须越过的强基线；正式协议、数据与模型池需据此重新审查，原在线12000次聚合继续暂缓。',bold=True,size=10)

 zoom=doc.settings.element.find(qn('w:zoom'))
 if zoom is not None:zoom.set(qn('w:percent'),'100')
 doc.save(report)
 provenance={'schema':'teacher_e1_docx_report_build_v1','report':report.name,'sha256':sha(report),'source_e1_sha256':sha(R/'results/e1/e1_diagnostics.json'),'source_audit_sha256':sha(A/'audit_metrics.json'),'source_input_audit_sha256':sha(A/'input_reconstruction.json'),'source_physical_summary_sha256':sha(A/'physical_call_summary.json'),'planned_sections':11,'facts_checked':{'actual_e1_calls':8800,'strict_gsm_score':95,'same_candidate_vote_correct':165,'same_candidate_fused_correct':153,'fixed_correct_candidate_lost':33,'fixed_independent_repair':2},'extra_diagnostic_status':'COMPLETE_600_SEPARATELY_RECONCILED_AND_SCORED','new_model_calls_by_report_builder':0,'created_at_unix':time.time(),'render_and_visual_qa':'PENDING'}
 (OUT/'REPORT_BUILD.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'report':str(report),'bytes':report.stat().st_size,'source_verified':True,'planned_sections':11},ensure_ascii=False))
if __name__=='__main__':main()
