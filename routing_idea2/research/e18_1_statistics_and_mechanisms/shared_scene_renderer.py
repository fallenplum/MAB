"""Author the E18 mechanism figure from one editable geometric scene.

The native draw.io CLI is probed and recorded. SVG and PNG use local scene
renderers, with their actual provenance recorded even when a CLI is present.
No remote image, new package or experimental reward file is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import PIL
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1800, 1400
INK, MUTED, BLUE = "#162B43", "#506276", "#225E8F"
BORDER, PANEL, WHITE = "#B7C8D8", "#F4F7FA", "#FFFFFF"
FONT_FAMILY = "Arial"
FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


@dataclass
class Box:
    name: str
    x: float
    y: float
    width: float
    height: float
    fill: str = WHITE
    stroke: str = BORDER
    radius: float = 10
    parent: str = "root"


@dataclass
class Label:
    name: str
    x: float
    y: float
    width: float
    height: float
    text: str
    size: int = 22
    bold: bool = False
    color: str = INK
    align: str = "center"
    parent: str = "root"


@dataclass
class Arrow:
    name: str
    source: str
    target: str
    points: list
    parent: str


def build_scene():
    boxes, labels, arrows = [], [], []

    def label(name, x, y, w, h, text, size=22, bold=False, color=INK, align="center", parent="root"):
        labels.append(Label(name, x, y, w, h, text, size, bold, color, align, parent))

    def box(name, x, y, w, h, fill=WHITE, parent="root"):
        boxes.append(Box(name, x, y, w, h, fill, BORDER, 10, parent))

    def card(name, x, y, w, h, heading, body, parent, heading_size=24):
        box(name, x, y, w, h, parent=parent)
        label(name+"_title", x+10, y+16, w-20, 34, heading, heading_size, True, parent=name)
        for index, line in enumerate(body):
            label(name+f"_line{index}", x+10, y+62+30*index, w-20, 28, line, 21, color=MUTED, parent=name)

    label("title", 60, 30, 1680, 56, "Controlled audits of ensemble-router supervision", 40, True, align="left")
    label("setting", 60, 94, 1680, 40,
          "Six frozen LMs  |  K = 3  |  20 teams  |  Frozen V2 aggregator  |  Cached outcomes",
          24, color=MUTED, align="left")

    box("training", 60, 170, 1680, 350, PANEL)
    label("training_header", 90, 190, 1600, 42, "Training-label intervention", 29, True, BLUE, "left", "training")
    card("cache", 90, 260, 240, 140, "Cached feedback", ["20 teams × 3 repeats", "600 original questions"], "training")
    card("ties", 430, 260, 240, 140, "Empirical ties", ["462–467 / 480", "multiple-best per fold"], "training")
    card("champion", 770, 260, 240, 140, "Champion rule", ["100 frozen action", "renumberings"], "training")
    card("head", 1110, 260, 240, 140, "Shared linear head", ["340 parameters", "Frozen features + L2"], "training", 22)
    card("choice", 1450, 260, 240, 140, "Physical team", ["Map back to fixed", "team identities"], "training")
    for index, (source, target) in enumerate(zip(("cache", "ties", "champion", "head"), ("ties", "champion", "head", "choice"))):
        first = next(b for b in boxes if b.name == source)
        second = next(b for b in boxes if b.name == target)
        arrows.append(Arrow("train_arrow"+str(index), source, target,
                            [[first.x+first.width, 330], [second.x, 330]], "training"))
    label("training_metric", 90, 425, 1600, 42,
          "Fixed semantic_exact   ·   SPO+ (M4)  6.135%   ·   Ridge / TSPO+  100%",
          28, True, BLUE, parent="training")
    label("training_metric_scope", 90, 470, 1600, 28,
          "Mean physical-team identity consistency across 100 renumberings", 21, color=MUTED, parent="training")

    box("inference", 60, 550, 1680, 300, PANEL)
    label("inference_header", 90, 570, 1600, 42, "Training symmetry and inference priority", 29, True, BLUE, "left", "inference")
    card("uniform", 90, 640, 330, 130, "Uniform tied champions", ["TSPO+ training target"], "inference", 24)
    card("small_gap", 560, 640, 480, 130, "397 / 600 near ties", ["Top-score gap ≤ 1e-8"], "inference", 28)
    card("numeric", 1190, 640, 500, 130, "37.308% identity consistency", ["numeric_near inference priority"], "inference", 26)
    arrows.extend([
        Arrow("inference_arrow0", "uniform", "small_gap", [[420, 705], [560, 705]], "inference"),
        Arrow("inference_arrow1", "small_gap", "numeric", [[1040, 705], [1190, 705]], "inference"),
    ])
    label("inference_scope", 90, 790, 1600, 30,
          "Threshold fixed before evaluation. Semantic near-tie priority retains 100% identity consistency.",
          21, color=MUTED, parent="inference")

    box("order", 60, 880, 1680, 390, PANEL)
    label("order_header", 90, 900, 1600, 42, "Order of supervision operators", 29, True, BLUE, "left", "order")
    card("draws", 90, 990, 300, 150, "Same 8 joint draws", ["Same features, capacity", "and regularization"], "order", 25)
    box("loss_then_mean", 590, 970, 510, 80, parent="order")
    label("loss_then_mean_label", 605, 990, 480, 40, "LossThenMean   E[L(q, R)]", 27, True, parent="loss_then_mean")
    box("mean_then_loss", 590, 1090, 510, 80, parent="order")
    label("mean_then_loss_label", 605, 1110, 480, 40, "MeanThenLoss   L(q, E[R])", 27, True, parent="mean_then_loss")
    box("loss_accuracy", 1260, 970, 430, 80, parent="order")
    label("loss_accuracy_label", 1270, 985, 410, 50, "86.000%", 36, True, BLUE, parent="loss_accuracy")
    box("mean_accuracy", 1260, 1090, 430, 80, parent="order")
    label("mean_accuracy_label", 1270, 1105, 410, 50, "85.111%", 36, True, BLUE, parent="mean_accuracy")
    arrows.extend([
        Arrow("order_upper", "draws", "loss_then_mean", [[390, 1030], [490, 1030], [490, 1010], [590, 1010]], "root"),
        Arrow("order_lower", "draws", "mean_then_loss", [[390, 1100], [490, 1100], [490, 1130], [590, 1130]], "root"),
        Arrow("order_upper_result", "loss_then_mean", "loss_accuracy", [[1100, 1010], [1260, 1010]], "order"),
        Arrow("order_lower_result", "mean_then_loss", "mean_accuracy", [[1100, 1130], [1260, 1130]], "order"),
    ])
    label("order_delta", 1260, 1180, 430, 34, "Δ = −0.889 percentage points", 22, True, parent="order")
    label("truth", 90, 1180, 1120, 30, "Known-truth control   μ = (0.54, 0.55)", 23, True, align="left", parent="order")
    label("truth_scope", 90, 1220, 1120, 30, "Synthetic example changes the optimum with exact means.", 21, color=MUTED, align="left", parent="order")

    label("evidence_scope", 60, 1300, 1680, 34,
          "Accuracies shown use reused development data. Independent fresh400 accuracy remains pending.",
          22, True)
    label("replicate_scope", 60, 1346, 1680, 30,
          "100 renumberings reuse the same 600 questions; three generation repeats remain within-question measurements.",
          21, color=MUTED)
    return boxes, labels, arrows


def font(size, bold=False, scale=1):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), round(size*scale))


def validate_geometry(boxes, labels, arrows):
    all_names = [item.name for collection in (boxes, labels, arrows) for item in collection]
    if len(all_names) != len(set(all_names)):
        raise AssertionError("Duplicate scene identity")
    box_map = {box.name: box for box in boxes}
    widths = []
    for item in boxes + labels:
        if item.x < 0 or item.y < 0 or item.x+item.width > WIDTH or item.y+item.height > HEIGHT:
            raise AssertionError("Off-canvas geometry " + item.name)
        if item.parent != "root":
            parent = box_map[item.parent]
            if item.x < parent.x or item.y < parent.y or item.x+item.width > parent.x+parent.width or item.y+item.height > parent.y+parent.height:
                raise AssertionError("Child outside parent " + item.name)
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for item in labels:
        text_box = draw.textbbox((0, 0), item.text, font=font(item.size, item.bold))
        actual_width, actual_height = text_box[2]-text_box[0], text_box[3]-text_box[1]
        if actual_width > item.width or actual_height > item.height:
            raise AssertionError("Clipped label " + item.name + " " + str((actual_width, actual_height)))
        widths.append(actual_width/item.width)
    for arrow in arrows:
        if arrow.source not in box_map or arrow.target not in box_map:
            raise AssertionError("Dangling arrow")
        for start, end in zip(arrow.points, arrow.points[1:]):
            if start[0] != end[0] and start[1] != end[1]:
                raise AssertionError("Non-orthogonal connector")
        if math.dist(arrow.points[-2], arrow.points[-1]) < 20:
            raise AssertionError("Short arrowhead segment")
    return dict(status="PASS",editable_boxes=len(boxes),editable_labels=len(labels),connectors=len(arrows),
                maximum_label_width_fraction=max(widths),all_labels_fit=True,all_children_contained=True,
                all_connectors_have_valid_endpoints=True)


def drawio(boxes, labels, arrows):
    document = ET.Element("mxfile", host="local", version="26.0.0", type="device")
    diagram = ET.SubElement(document, "diagram", name="E18 mechanism audit", id="e18-mechanism")
    model = ET.SubElement(diagram, "mxGraphModel", dx=str(WIDTH), dy=str(HEIGHT), grid="1", gridSize="10",
                          page="1", pageScale="1", pageWidth=str(WIDTH), pageHeight=str(HEIGHT), background=WHITE)
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    identities = {item.name: str(index+2) for index,item in enumerate(boxes+labels+arrows)}
    box_map = {box.name: box for box in boxes}
    def parent_id(name):
        return "1" if name == "root" else identities[name]
    for box in boxes:
        parent = box_map.get(box.parent)
        style = f"rounded=1;arcSize=10;html=1;fillColor={box.fill};strokeColor={box.stroke};strokeWidth=2;container=1;pointerEvents=0;"
        cell = ET.SubElement(root,"mxCell",id=identities[box.name],value="",style=style,vertex="1",parent=parent_id(box.parent))
        ET.SubElement(cell,"mxGeometry",x=str(box.x-(parent.x if parent else 0)),y=str(box.y-(parent.y if parent else 0)),
                      width=str(box.width),height=str(box.height),attrib={"as":"geometry"})
    for label in labels:
        parent=box_map.get(label.parent)
        style=f"text;html=1;whiteSpace=wrap;overflow=hidden;fillColor=none;strokeColor=none;fontFamily={FONT_FAMILY};fontSize={label.size};fontColor={label.color};fontStyle={1 if label.bold else 0};align={label.align};verticalAlign=middle;spacing=0;"
        cell=ET.SubElement(root,"mxCell",id=identities[label.name],value=label.text,style=style,vertex="1",parent=parent_id(label.parent))
        ET.SubElement(cell,"mxGeometry",x=str(label.x-(parent.x if parent else 0)),y=str(label.y-(parent.y if parent else 0)),
                      width=str(label.width),height=str(label.height),attrib={"as":"geometry"})
    for arrow in arrows:
        source,target=box_map[arrow.source],box_map[arrow.target]
        first,last=arrow.points[0],arrow.points[-1]
        style=("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
               f"strokeColor={BLUE};strokeWidth=2;endArrow=block;endFill=1;"
               f"exitX={(first[0]-source.x)/source.width};exitY={(first[1]-source.y)/source.height};"
               f"entryX={(last[0]-target.x)/target.width};entryY={(last[1]-target.y)/target.height};")
        cell=ET.SubElement(root,"mxCell",id=identities[arrow.name],value="",style=style,edge="1",parent=parent_id(arrow.parent),
                           source=identities[arrow.source],target=identities[arrow.target])
        geometry=ET.SubElement(cell,"mxGeometry",relative="1",attrib={"as":"geometry"})
        if len(arrow.points)>2:
            points=ET.SubElement(geometry,"Array",attrib={"as":"points"})
            parent=box_map.get(arrow.parent)
            for x,y in arrow.points[1:-1]:
                ET.SubElement(points,"mxPoint",x=str(x-(parent.x if parent else 0)),y=str(y-(parent.y if parent else 0)))
    ET.indent(document,space="  ")
    return ET.tostring(document,encoding="utf-8",xml_declaration=True)


def svg(boxes, labels, arrows):
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
           '<title>E18 controlled mechanism audits of ensemble-router supervision</title>',
           '<desc>Three controlled audits separate training-label conventions, inference ties, and supervision operator order. Development accuracy only; independent fresh400 accuracy pending.</desc>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{WHITE}"/>',
           f'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="{BLUE}"/></marker></defs>']
    for box in boxes:
        lines.append(f'<rect id="{box.name}" x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="{box.radius}" fill="{box.fill}" stroke="{box.stroke}" stroke-width="2"/>')
    for arrow in arrows:
        points=" ".join(f"{x},{y}" for x,y in arrow.points)
        lines.append(f'<polyline id="{arrow.name}" points="{points}" fill="none" stroke="{BLUE}" stroke-width="2" marker-end="url(#arrow)"/>')
    for label in labels:
        x=label.x if label.align=="left" else label.x+label.width/2
        anchor="start" if label.align=="left" else "middle"
        weight="700" if label.bold else "400"
        lines.append(f'<text id="{label.name}" x="{x}" y="{label.y+label.height/2}" dominant-baseline="central" text-anchor="{anchor}" font-family="{FONT_FAMILY},sans-serif" font-size="{label.size}" font-weight="{weight}" fill="{label.color}">{html.escape(label.text)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)+"\n"


def png(boxes, labels, arrows, path, scale=1):
    image=Image.new("RGB",(round(WIDTH*scale),round(HEIGHT*scale)),WHITE)
    draw=ImageDraw.Draw(image)
    for box in boxes:
        bounds=tuple(round(value*scale) for value in (box.x,box.y,box.x+box.width,box.y+box.height))
        draw.rounded_rectangle(bounds,radius=round(box.radius*scale),fill=box.fill,outline=box.stroke,width=round(2*scale))
    for arrow in arrows:
        points=[(round(x*scale),round(y*scale)) for x,y in arrow.points]
        draw.line(points,fill=BLUE,width=round(2*scale),joint="curve")
        start,end=points[-2],points[-1]
        theta=math.atan2(end[1]-start[1],end[0]-start[0])
        length,half_width=10*scale,5*scale
        base=(end[0]-length*math.cos(theta),end[1]-length*math.sin(theta))
        draw.polygon([end,(base[0]-half_width*math.sin(theta),base[1]+half_width*math.cos(theta)),
                      (base[0]+half_width*math.sin(theta),base[1]-half_width*math.cos(theta))],fill=BLUE)
    for label in labels:
        x=label.x if label.align=="left" else label.x+label.width/2
        draw.text((round(x*scale),round((label.y+label.height/2)*scale)),label.text,
                  font=font(label.size,label.bold,scale),fill=label.color,anchor="lm" if label.align=="left" else "mm")
    image.save(path,dpi=(300,300))


def cli_probe():
    paths=[Path("C:/Program Files/draw.io/draw.io.exe"),Path("C:/Program Files/draw.io/drawio.exe"),
           Path("C:/Program Files (x86)/draw.io/draw.io.exe")]
    if os.environ.get("LOCALAPPDATA"):
        paths.append(Path(os.environ["LOCALAPPDATA"])/"Programs/draw.io/draw.io.exe")
    return dict(path_binary=shutil.which("draw.io") or shutil.which("drawio"),
                common_paths=[dict(path=str(path),exists=path.exists()) for path in paths])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("figures"))
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    boxes,labels,arrows=build_scene()
    qa=validate_geometry(boxes,labels,arrows)
    stem="e18_mechanism_audit"
    paths={extension:args.output_dir/f"{stem}.{extension}" for extension in ("drawio","svg","png","geometry.json")}
    qa_path=args.output_dir/"MECHANISM_DIAGRAM_QA.json"
    readme=args.output_dir/"README.md"
    if not args.force and any(path.exists() for path in [*paths.values(),qa_path,readme]):
        raise FileExistsError("Existing diagram outputs require --force")
    xml=drawio(boxes,labels,arrows)
    parsed=ET.fromstring(xml)
    ids=[cell.attrib["id"] for cell in parsed.iter("mxCell")]
    if len(ids)!=len(set(ids)) or not {"0","1"}.issubset(ids):raise AssertionError("Invalid draw.io IDs")
    if any(cell.find("mxGeometry") is None for cell in parsed.iter("mxCell") if cell.get("edge")=="1"):
        raise AssertionError("Missing connector geometry")
    paths["drawio"].write_bytes(xml)
    vector=svg(boxes,labels,arrows)
    ET.fromstring(vector)
    paths["svg"].write_text(vector,encoding="utf-8")
    png(boxes,labels,arrows,paths["png"])
    Image.open(paths["png"]).verify()
    scene=dict(width=WIDTH,height=HEIGHT,font_family=FONT_FAMILY,boxes=[asdict(v) for v in boxes],
               labels=[asdict(v) for v in labels],arrows=[asdict(v) for v in arrows])
    paths["geometry.json"].write_text(json.dumps(scene,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    probe=cli_probe()
    qa.update(canvas=[WIDTH,HEIGHT],font_family=FONT_FAMILY,font_files=[str(FONT_REGULAR),str(FONT_BOLD)],
              native_drawio_cli_probe=probe,native_cli_export_used=False,
              native_cli_status="UNAVAILABLE" if not probe["path_binary"] and not any(p["exists"] for p in probe["common_paths"]) else "AVAILABLE_NOT_USED",
              svg_renderer="Local SVG writer from shared editable scene",
              png_renderer=f"Pillow {PIL.__version__} from the same shared scene",pixel_size=[WIDTH,HEIGHT],
              visual_review="PENDING_ROOT_VISUAL_QA",new_physical_llm_calls=0,
              files={key:dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key,path in paths.items()},
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    qa_path.write_text(json.dumps(qa,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    readme.write_text("""# E18 mechanism figure

The editable `.drawio`, vector `.svg`, and 1800 × 1400 PNG share the geometry in
`e18_mechanism_audit.geometry.json`. The source is `../make_mechanism_diagram.py`.

The native draw.io CLI was unavailable at the probed host locations and on PATH.
SVG was authored locally from the scene. PNG was rendered locally with Pillow.
These are local fallback renders. No native CLI export was performed. The PNG
is a standalone preview and contains no embedded draw.io XML. The `.drawio`
source preserves individual editable boxes, labels, connectors and containers.

Reproduce with `python -B -X utf8 make_mechanism_diagram.py` from the parent
directory. Add `--force` to regenerate these figure outputs during editing.

All displayed measurements are aggregate E17 development results. Independent
fresh400 accuracy remains pending. The figure contains no question text,
answers, private question identifiers, model weights or new LLM calls.
""",encoding="utf-8")
    print(json.dumps(dict(status=qa["status"],native_cli_status=qa["native_cli_status"],size=qa["pixel_size"],files=[str(path) for path in paths.values()]),ensure_ascii=False))


if __name__=="__main__":
    main()
