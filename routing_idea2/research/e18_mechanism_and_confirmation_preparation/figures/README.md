# E18 mechanism figure

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
