"""Local editable scene renderer copied from the frozen E18 geometric helper.

Selected rendering functions retain their previous implementation. Titles and
XML page metadata are generic in this independent E19 copy.
Source SHA256 9bd5c45bd2f86d98c837fa844403e727b11a2073e0795ec3b3f22a3cbb3b2cf8.
"""
from __future__ import annotations
from dataclasses import dataclass
import html
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
    diagram = ET.SubElement(document, "diagram", name="Editable evidence scene", id="evidence-scene")
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
           '<title>Editable ensemble-router evidence scene</title>',
           '<desc>An editable scene composed of labeled evidence boxes and directed connectors.</desc>',
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
