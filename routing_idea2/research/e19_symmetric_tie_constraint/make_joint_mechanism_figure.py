"""Render the E17/E18/E19 evidence scene from the completed public summary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import shared_scene_renderer as scene


ROOT=Path(__file__).resolve().parent


def build(summary):
    boxes,labels,arrows=[],[],[]

    def box(name,x,y,width,height,parent="root",fill=None):
        boxes.append(scene.Box(name,x,y,width,height,fill or scene.WHITE,scene.BORDER,10,parent))

    def label(name,x,y,width,height,text,size=22,bold=False,parent="root",align="center",color=None):
        labels.append(scene.Label(name,x,y,width,height,text,size,bold,color or scene.INK,align,parent))

    def card(name,x,y,width,height,title,body,parent,size=22):
        box(name,x,y,width,height,parent)
        label(name+"_title",x+10,y+14,width-20,34,title,size+1,True,name)
        for index,text in enumerate(body):
            label(name+"_line"+str(index),x+10,y+60+31*index,width-20,28,text,size,parent=name,color=scene.MUTED)

    def arrow(name,source,target,x1,y1,x2,y2,parent):
        arrows.append(scene.Arrow(name,source,target,[[x1,y1],[x2,y2]],parent))

    label("title",60,28,1680,55,"From tied rewards to training constraints",40,True,align="left")
    label("setting",60,92,1680,35,
          "Six frozen models  |  20 three-member teams  |  Fixed cached feedback  |  Shared linear head",
          24,align="left",color=scene.MUTED)
    box("history",60,160,1680,330,fill=scene.PANEL)
    label("history_header",90,180,1620,40,"Historical evidence separates behavior and quality",29,True,"history","left",scene.BLUE)
    card("renumber",90,250,430,180,"E17 champion intervention",
         ["100 renumberings on 600 items","SPO team agreement 6.135%","Ridge / uniform TSPO 100%"],"history")
    card("symmetry",690,250,430,180,"E17 inference control",
         ["TSPO 397 / 600 near ties","Semantic priority 100% agreement","Numeric near-tie priority 37.308%"],"history")
    card("quality",1290,250,420,180,"E18 frozen quality evaluation",
         ["2,998 scorable / 3,000 requested","TSPO − SPO +0.01175 pp","Corrected 377 / damaged 376"],"history",21)
    arrow("history_a", "renumber","symmetry",520,340,690,340,"history")
    arrow("history_b", "symmetry","quality",1120,340,1290,340,"history")
    label("history_scope",90,445,1620,28,
          "E18.1 marginal 95% interval [−0.667,+0.690] pp; post-result supplement on the frozen evaluation.",
          22,parent="history",color=scene.MUTED)

    box("theory",60,520,1680,340,fill=scene.PANEL)
    label("theory_header",90,540,1620,40,"Two constructions give different falsifiable predictions",29,True,"theory","left",scene.BLUE)
    card("a1",90,610,430,180,"A1 finite repeated feedback",
         ["t=(1−a)^L","d*=0 exactly when tp ≥ 1−t","Fixed L retains the same target risk"],"theory",21)
    card("a2",690,610,430,180,"A2 symmetric constraint strength",
         ["T truly tied; shared parameters","d_X2=(−46+51β)/600","Strict decision crosses at 46/51"],"theory",21)
    card("boundary",1290,610,420,180,"Explicit construction boundary",
         ["λ=60 belongs to this 3D example","Reward 1 → 0.95 at tie → 0.9","Real W penalty inherited by fold"],"theory",21)
    arrow("theory_b","a2","boundary",1120,700,1290,700,"theory")
    label("theory_scope",90,815,1620,28,
          "Directional champion preference and direction-neutral score contraction are distinct mechanisms.",
          22,parent="theory",color=scene.MUTED)

    box("pilot",60,890,1680,380,fill=scene.PANEL)
    label("pilot_header",90,910,1620,40,"E19 isolates the fully tied-row term with γ=0",29,True,"pilot","left",scene.BLUE)
    card("b",90,1000,300,170,"B retains constraint",
         ["β=1","Uniform empirical champions","Original denominator N=480"],"pilot",20)
    card("c",520,1000,300,170,"C disables tied-row term",
         ["β=0","Same labels on all other rows","Same original L2, including bias"],"pilot",20)
    card("qualification",950,1000,330,170,"P0 implementation gate",
         summary["qualification"],"pilot",20)
    card("results",1410,1000,300,170,"P1 uniform τ=1e−8",
         summary["results"],"pilot",20)
    label("endpoint_compare",410,1070,90,30,"versus",21,True,"pilot")
    arrows.append(scene.Arrow("pilot_a","b","qualification",
                              [[240,1000],[240,980],[1115,980],[1115,1000]],"root"))
    arrow("pilot_b","c","qualification",820,1085,950,1085,"pilot")
    arrow("pilot_c","qualification","results",1280,1085,1410,1085,"pilot")
    label("pilot_formula",90,1190,1620,34,
          "Strict-change contribution +1.056 pp (38 corrected / 19 damaged); numerical-tie region +0.386 pp",
          23,True,"pilot")
    label("raw_sensitivity",90,1230,1620,28,
          "Raw exact-argmax diagnostic +1.000 pp, 95% CI [−0.333,+2.333] pp; the numerical service definition matters.",
          21,parent="pilot",color=scene.MUTED)
    label("scope",60,1300,1680,34,
          "P0 + P1 completed with 10 main and 2 mapping fits on train600 OOF development; additional LLM calls 0.",
          22,True)
    label("footer",60,1346,1680,28,
          "Independent-confirmation design remains a later stage under a separate frozen protocol.",
          22,color=scene.MUTED)
    return boxes,labels,arrows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=ROOT/"figures")
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text(encoding="utf-8"))
    if set(summary)!={"qualification","results"}:
        raise ValueError("Exact public figure summary fields required")
    if any(not isinstance(summary[key],list) or len(summary[key])!=3
           or any(not isinstance(value,str) for value in summary[key]) for key in summary):
        raise ValueError("Each figure card requires three explicit public lines")
    boxes,labels,arrows=build(summary)
    qa=scene.validate_geometry(boxes,labels,arrows)
    output=args.output_root
    output.mkdir(parents=True,exist_ok=True)
    xml=ET.fromstring(scene.drawio(boxes,labels,arrows))
    xml.find("diagram").set("name","E17-E18-E19 constraint evidence")
    ET.indent(xml,space="  ")
    (output/"figure.drawio").write_bytes(ET.tostring(xml,encoding="utf-8",xml_declaration=True))
    (output/"figure.svg").write_text(scene.svg(boxes,labels,arrows),encoding="utf-8")
    scene.png(boxes,labels,arrows,output/"figure.png")
    scene.Image.open(output/"figure.png").verify()
    (output/"figure.geometry.json").write_text(json.dumps(dict(
        boxes=[asdict(value) for value in boxes],labels=[asdict(value) for value in labels],
        arrows=[asdict(value) for value in arrows]),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    probe=scene.cli_probe()
    qa.update(status="SCENE_RENDERED_FROM_PUBLIC_RESULT_SUMMARY",
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              shared_renderer_sha256=hashlib.sha256(Path(scene.__file__).read_bytes()).hexdigest(),
              native_cli_found=bool(probe["path_binary"]) or any(row["exists"] for row in probe["common_paths"]),
              native_cli_export_used=False,local_renderers="Shared scene SVG writer and Pillow",
              real_model_fits=0,new_llm_calls=0,experiment_reward_reads=0,bootstrap_runs=0)
    (output/"figure.qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(qa,ensure_ascii=True))


if __name__=="__main__":
    main()
