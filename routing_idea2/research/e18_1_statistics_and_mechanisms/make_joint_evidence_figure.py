"""Editable E17/E18 evidence diagram using the frozen shared scene renderer."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
RENDERER = ROOT / "shared_scene_renderer.py"
RENDERER_SHA = "9bd5c45bd2f86d98c837fa844403e727b11a2073e0795ec3b3f22a3cbb3b2cf8"
STATISTICS_SHA = "ab398af856c9daf14c93b4f35cb0eb407db522aad3c29457c0aea356bd8faefe"


def load_renderer():
    if hashlib.sha256(RENDERER.read_bytes()).hexdigest() != RENDERER_SHA:
        raise ValueError("Existing renderer source identity differs")
    spec = importlib.util.spec_from_file_location("e18_1_shared_scene_renderer", RENDERER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_scene(api):
    boxes, labels, arrows = [], [], []

    def label(name, x, y, width, height, text, size=22, bold=False,
              align="center", parent="root", color=None):
        labels.append(api.Label(name, x, y, width, height, text, size, bold,
                                color or api.INK, align, parent))

    def box(name, x, y, width, height, parent="root", fill=None):
        boxes.append(api.Box(name, x, y, width, height, fill or api.WHITE,
                             api.BORDER, 10, parent))

    def card(name, x, y, width, height, title, lines, parent, size=22):
        box(name, x, y, width, height, parent)
        label(name + "_title", x+10, y+14, width-20, 36, title, size+1, True, parent=name)
        for index, text in enumerate(lines):
            label(name + "_body" + str(index), x+10, y+60+31*index,
                  width-20, 29, text, size, parent=name, color=api.MUTED)

    def arrow(name, source, target, x1, y1, x2, y2, parent):
        arrows.append(api.Arrow(name, source, target, [[x1,y1],[x2,y2]], parent))

    label("title", 60, 25, 1680, 55, "When ties become preferences", 40, True, "left")
    label("subtitle", 60, 90, 1680, 35,
          "E17 controlled interventions and E18 frozen-policy evaluation", 25, align="left", color=api.MUTED)

    box("intervention", 60, 160, 1680, 390, fill=api.PANEL)
    label("intervention_title", 90, 178, 1620, 42,
          "Champion convention changes the trained router", 29, True, "left", "intervention", api.BLUE)
    card("ties", 90, 250, 300, 180, "Empirical supervision",
         ["20 teams × 3 repeats", "462–467 / 480", "multiple-best questions"], "intervention")
    card("training", 510, 250, 330, 180, "Training intervention",
         ["100 action renumberings", "Fixed semantic inference", "340-parameter shared head"], "intervention")
    card("identity", 960, 250, 340, 180, "Physical-team agreement",
         ["SPO_coordinate  6.135%", "Ridge / TSPO  100%", "Same 600 development items"], "intervention")
    card("inference", 1420, 250, 290, 180, "Inference-only control",
         ["TSPO: 397 near ties", "numeric_near  37.308%", "semantic_near  100%"], "intervention", 20)
    arrow("a1", "ties", "training", 390, 340, 510, 340, "intervention")
    arrow("a2", "training", "identity", 840, 340, 960, 340, "intervention")
    arrow("a3", "identity", "inference", 1300, 340, 1420, 340, "intervention")
    label("loss_difference", 90, 452, 1620, 37,
          "Tied champions s,t:   L_t − L_s = 2(q_s − q_t)   →   different shared-parameter gradients",
          24, True, parent="intervention")
    label("intervention_scope", 90, 503, 1620, 27,
          "Training labels and score-space tie priority are separately controlled.",
          21, parent="intervention", color=api.MUTED)

    box("operator", 60, 580, 1680, 300, fill=api.PANEL)
    label("operator_title", 90, 598, 1620, 42,
          "Symmetry and the order of averaging need separate checks", 29, True, "left", "operator", api.BLUE)
    card("same_draws", 90, 670, 310, 150, "Same 8 joint draws",
         ["Same features and L2", "Reward samples fixed"], "operator")
    card("operators", 540, 670, 450, 150, "Two supervision operators",
         ["LossThenMean  E[L(q,R)]", "MeanThenLoss  L(q,E[R])"], "operator")
    card("operator_results", 1130, 670, 580, 150, "MeanThenLoss had lower development accuracy",
         ["MeanThenLoss − LossThenMean  −0.889 pp", "Known-truth example has different optima"], "operator", 22)
    arrow("b1", "same_draws", "operators", 400, 745, 540, 745, "operator")
    arrow("b2", "operators", "operator_results", 990, 745, 1130, 745, "operator")

    box("confirmation", 60, 910, 1680, 360, fill=api.PANEL)
    label("confirmation_title", 90, 928, 1620, 42,
          "Frozen-policy quality evaluation gives a near-zero net gain", 29, True, "left", "confirmation", api.BLUE)
    card("evaluation", 90, 1000, 480, 210, "Requested 3,000  /  evaluable 2,998",
         ["Ridge  86.6637%", "SPO_coordinate  86.3961%",
          "TSPO_uniform  86.4079%", "GlobalFixed  86.8634%"], "confirmation", 22)
    card("gain", 710, 1000, 420, 210, "TSPO − SPO  +0.01175 pp",
         ["95% CI  [−0.667, +0.690] pp", "Corrected 377  /  damaged 376",
          "Task directions oppose", "Seed gains: +0.201 / +0.268 / −0.433 pp"], "confirmation", 20)
    card("boundary", 1270, 1000, 440, 210, "Retained limitations",
         ["2 input-reference incompatibilities", "Excluded outcomes stay unknown",
          "Full execution costs retained", "E18.1 is a post-result supplement"], "confirmation", 21)
    arrow("c1", "evaluation", "gain", 570, 1105, 710, 1105, "confirmation")
    arrow("c2", "gain", "boundary", 1130, 1105, 1270, 1105, "confirmation")
    label("replication", 60, 1300, 1680, 34,
          "100 renumberings reuse 600 items; three generation repeats remain within-question measurements.",
          23, True)
    label("interpretation", 60, 1345, 1680, 30,
          "E18.1: 20,000 stratified question bootstraps; marginal intervals; specified after viewing E18 results.",
          22, color=api.MUTED)
    return boxes, labels, arrows


def main():
    api = load_renderer()
    boxes, labels, arrows = build_scene(api)
    qa = api.validate_geometry(boxes, labels, arrows)
    xml = api.drawio(boxes, labels, arrows)
    document = ET.fromstring(xml)
    document.find("diagram").set("name", "E17-E18 joint evidence")
    ET.indent(document, space="  ")
    (ROOT / "figure.drawio").write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True))
    vector = api.svg(boxes, labels, arrows)
    vector = vector.replace("E18 controlled mechanism audits of ensemble-router supervision", "E17 and E18 joint supervision-distortion evidence")
    vector = vector.replace("Three controlled audits separate training-label conventions, inference ties, and supervision operator order. Development accuracy only; independent fresh400 accuracy pending.", "Controlled development interventions separate champion labels, inference ties and operator order. Frozen E18 policies are evaluated on2998 valid independent questions with2 excluded inputs.")
    (ROOT / "figure.svg").write_text(vector, encoding="utf-8")
    api.png(boxes, labels, arrows, ROOT / "figure.png")
    api.Image.open(ROOT / "figure.png").verify()
    geometry = dict(width=api.WIDTH, height=api.HEIGHT,
                    boxes=[asdict(value) for value in boxes],
                    labels=[asdict(value) for value in labels],
                    arrows=[asdict(value) for value in arrows])
    (ROOT / "figure.geometry.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    probe = api.cli_probe()
    safe_probe = dict(path_binary_found=bool(probe["path_binary"]),
                      common_candidates_checked=len(probe["common_paths"]),
                      common_candidates_found=sum(row["exists"] for row in probe["common_paths"]))
    qa.update(native_cli_probe=safe_probe, native_cli_export_used=False,
              svg_and_png_source="Shared editable scene with existing local SVG and Pillow renderers",
              existing_renderer_sha256=RENDERER_SHA,
              statistics_source_sha256=STATISTICS_SHA,
              bootstrap_resamples_by_root=20000,
              inference="Marginal percentile intervals; specified after E18 outcome inspection",
              figure_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              evidence="Existing E17 frozen summaries and completed E18 evaluable point results",
              new_training_runs=0, new_llm_calls=0, bootstrap_runs_by_figure=0)
    (ROOT / "figure.qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=True))


if __name__ == "__main__":
    main()
