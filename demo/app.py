"""
Demo: upload a satellite image and see how likely each label is.

Run from the project root:
    python demo/app.py
Then open http://localhost:7860
"""

import json
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import extract_features  # noqa: E402

MODEL_DIR = ROOT / "models"
EXAMPLES_DIR = ROOT / "demo" / "examples"

# key, name, plain description, colour (Pantone palette)
LABELS = [
    ("primary", "Primary forest", "Dense, untouched rainforest canopy", "#4A6175"),
    ("water", "Water", "Rivers and lakes", "#7BB3D9"),
    ("agriculture", "Agriculture", "Fields, pastures and cleared land", "#D6CF95"),
    ("road", "Road", "Roads and dirt tracks", "#9C9492"),
    ("cloudy", "Cloudy", "Image fully covered by clouds", "#A893AE"),
]

EXAMPLES = [
    ("train_23477.jpg", "Forest"),
    ("train_11173.jpg", "River"),
    ("train_28412.jpg", "Fields and tracks"),
    ("train_6545.jpg", "Town by a river"),
    ("train_23634.jpg", "Clouds"),
]

print("Loading models...")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
iso = joblib.load(MODEL_DIR / "isolation_forest.joblib")
rf = joblib.load(MODEL_DIR / "rf_balanced.joblib")
config = json.loads((MODEL_DIR / "config.json").read_text())
THRESHOLDS = config["thresholds"]
ANOMALY_LIMIT = config["anomaly_limit"]
USES_ANOMALY = "anomaly_score" in config["features"]
N_IMAGE_FEATURES = len(config["features"]) - USES_ANOMALY  # model may use only the first 18
print("Models loaded.")


def predict_probs(image):
    """Return (probability per label, anomaly score) for a PIL image."""
    feat = scaler.transform(extract_features(image).reshape(1, -1))
    anomaly = float(-iso.score_samples(feat)[0])
    feat = feat[:, :N_IMAGE_FEATURES]
    if USES_ANOMALY:
        feat = np.hstack([feat, [[anomaly]]])
    probs = {key: float(est.predict_proba(feat)[0][1])
             for (key, *_), est in zip(LABELS, rf.estimators_)}
    return probs, anomaly


EMPTY = """
<div class="results empty">
  <p>Upload an image or pick one of the examples to see what the model finds in it.</p>
</div>
"""


def render(probs, anomaly):
    rows = []
    for key, name, desc, color in LABELS:
        p, t = probs[key], THRESHOLDS[key]
        present = p >= t
        rows.append(f"""
        <div class="row {'on' if present else 'off'}" style="--c:{color}">
          <div class="row-head">
            <div class="row-name">
              <span class="name">{name}</span>
              <span class="desc">{desc}</span>
            </div>
            <span class="pct">{round(p * 100)}<small>%</small></span>
          </div>
          <div class="bar" role="img" aria-label="{name}: {round(p * 100)} percent, threshold {round(t * 100)} percent">
            <div class="fill" style="width:{p * 100:.1f}%"></div>
            <div class="tick" style="left:{t * 100:.1f}%"></div>
          </div>
          <div class="verdict">{'Present' if present else 'Not present'}<span>, needs {round(t * 100)}%</span></div>
        </div>""")

    found = [name.lower() for key, name, *_ in LABELS if probs[key] >= THRESHOLDS[key]]
    if not found:
        summary = "Nothing reaches its threshold."
    elif len(found) == 1:
        summary = f"Found: {found[0]}."
    else:
        summary = "Found: " + ", ".join(found[:-1]) + " and " + found[-1] + "."
    note = ""
    if anomaly > ANOMALY_LIMIT:
        note = ("<p class='warn'>This image looks very different from the Amazon satellite "
                "images the model learned from, so these numbers are unreliable.</p>")
    return f"""
    <div class="results">
      <p class="summary">{summary}</p>
      {note}
      {''.join(rows)}
    </div>"""


def classify(image):
    if image is None:
        return EMPTY
    return render(*predict_probs(image))


CSS = """
.gradio-container {
  --vista: #C6E3C6;
  --mist-bg: #EAF1E8;
  --paper: #F7FAF5;
  --ink: #2F4253;
  --fusion: #4A6175;
  --hematite: #6F6A64;
  --cloud: #9C9492;
  --track: #DCE4D9;
}
body, .gradio-container, .main, .contain { background: #EAF1E8 !important; }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; padding: 40px 24px 56px !important;
  font-family: 'Bricolage Grotesque', system-ui, sans-serif !important; color: var(--ink); }
footer { display: none !important; }

#hero h1 { font-size: clamp(2.6rem, 6.4vw, 5.2rem); line-height: .92; margin: 0 0 18px;
  font-weight: 800; font-variation-settings: "wdth" 75, "opsz" 96;
  letter-spacing: -0.02em; color: var(--fusion); max-width: 15ch; }
#hero p { font-size: 1.12rem; line-height: 1.5; color: var(--hematite); max-width: 60ch; margin: 0; }
#hero { margin-bottom: 28px; padding-bottom: 28px; border-bottom: 2px solid var(--vista); }


.results { display: flex; flex-direction: column; gap: 18px; }
.results.empty { min-height: 300px; justify-content: center; }
.results.empty p { color: var(--hematite); font-size: 1.05rem; max-width: 34ch; line-height: 1.5; }
.summary { font-size: 1.35rem; font-weight: 700; color: var(--fusion); margin: 0 0 4px; line-height: 1.25; }

.row-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; }
.row-name { display: flex; flex-direction: column; }
.name { font-weight: 700; font-size: 1.08rem; color: var(--ink); }
.desc { font-size: .88rem; color: var(--hematite); }
.pct { font-size: 2.3rem; font-weight: 800; line-height: 1; color: var(--ink);
  font-variant-numeric: tabular-nums; font-variation-settings: "wdth" 75; }
.pct small { font-size: 1rem; font-weight: 600; margin-left: 1px; }
.row.off .pct, .row.off .name { color: var(--cloud); }

.bar { position: relative; height: 16px; margin-top: 8px; background: var(--track); border-radius: 999px; }
.fill { position: absolute; top: 0; bottom: 0; left: 0; background: var(--c); border-radius: 999px;
  transition: width .6s cubic-bezier(.2,.8,.2,1); }
.row.off .fill { opacity: .45; }
.tick { position: absolute; top: -6px; bottom: -6px; width: 4px; margin-left: -2px;
  background: var(--ink); border-radius: 2px; box-shadow: 0 0 0 2px #EAF1E8; }
.verdict { margin-top: 6px; font-size: .85rem; font-weight: 600; color: var(--fusion); }
.verdict span { font-weight: 400; color: var(--hematite); }
.row.off .verdict { color: var(--cloud); }

.warn { margin: 0 0 4px; padding: 12px 14px; border-radius: 12px; background: #EFE9F1;
  border-left: 4px solid #A893AE; color: var(--ink); font-size: .95rem; line-height: 1.45; }

@media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..800&display=swap" rel="stylesheet">
"""

THEME = gr.themes.Base(
    font=[gr.themes.Font("Bricolage Grotesque"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#EAF1E8",
    block_background_fill="#F7FAF5",
    block_border_width="0px",
    block_label_background_fill="#F7FAF5",
    block_label_text_color="#6F6A64",
    body_text_color="#2F4253",
    color_accent="#4A6175",
)

with gr.Blocks(title="Amazon Satellite Image Classifier") as demo:
    gr.HTML(
        """<div id="hero">
          <h1>The Amazon, read from above</h1>
          <p>Upload a Planet satellite image of the rainforest. A Random Forest looks at its
          colours, textures and edges and estimates how likely each of five things is to be in it.
          One image can contain several at once.</p>
        </div>"""
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(type="pil", image_mode=None, label="Satellite image",
                             elem_id="upload", height=340, sources=["upload", "clipboard"])
            gr.Examples(
                examples=[[str(EXAMPLES_DIR / f)] for f, _ in EXAMPLES],
                example_labels=[label for _, label in EXAMPLES],
                inputs=image,
                label="Or try an image from the test set",
            )
        with gr.Column(scale=6):
            result = gr.HTML(EMPTY)

    image.change(classify, inputs=image, outputs=result)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
