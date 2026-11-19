"""
Flask app serving a scikit-learn LogisticRegression (3 classes: High / Low / Medium).
Deploy on Render:  Build: pip install -r requirements.txt   Start: gunicorn app:app
Files needed next to this one: logistic.pkl  (rename your f72c49f6-logistic.pkl)
Optional:                      scaler.pkl    (only if you scaled features in training)
"""
import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, os.environ.get("MODEL_FILE", "logistic.pkl"))
SCALER_PATH = os.path.join(BASE, "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

# Exact column order the model was trained on (read from the pickle).
FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]  # ['High', 'Low', 'Medium']

# ----------------------------------------------------------------------------
# >>> EDIT THIS BLOCK TO MATCH YOUR TRAINING ENCODING <<<
# The pickle holds only the model, not the encoders, so the number each category
# was turned into during training must be repeated here. The position in each
# list = the integer code used in training (LabelEncoder = alphabetical order,
# or whatever order you used for ordinal columns).
# ----------------------------------------------------------------------------
CATEGORICAL = {
    "Major_Category": ["Arts", "Business", "Engineering", "Health", "Science", "Social Science"],
    "Primary_Use_Case": ["Coding", "Brainstorming", "Research", "Writing"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
    "Paid_Subscription": ["No", "Yes"],
    "Institutional_Policy": ["Banned", "Restricted", "Allowed"],
}
NUMERIC = {  # label, min, max, step, default
    "Year_of_Study": ("Year of study", 1, 6, 1, 2),
    "Pre_Semester_GPA": ("Pre-semester GPA", 0, 10, 0.01, 3.0),
    "Weekly_GenAI_Hours": ("Weekly GenAI hours", 0, 80, 0.5, 8),
    "Traditional_Study_Hours": ("Traditional study hours / week", 0, 80, 0.5, 12),
    "Perceived_AI_Dependency": ("Perceived AI dependency (1-10)", 0, 10, 0.1, 5),
    "Skill_Retention_Score": ("Skill retention score", 0, 100, 0.5, 60),
}
# ----------------------------------------------------------------------------


def build_vector(data: dict) -> np.ndarray:
    row = []
    for name in FEATURES:
        if name not in data or data[name] in ("", None):
            raise ValueError(f"Missing value for '{name}'")
        val = data[name]
        if name in CATEGORICAL:
            opts = CATEGORICAL[name]
            if val not in opts:
                raise ValueError(f"'{val}' is not a valid option for '{name}'")
            row.append(float(opts.index(val)))
        else:
            row.append(float(val))
    X = np.array([row], dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    return X


def predict(data: dict) -> dict:
    X = build_vector(data)
    proba = model.predict_proba(X)[0]
    best = int(np.argmax(proba))
    return {
        "prediction": CLASSES[best],
        "confidence": round(float(proba[best]) * 100, 2),
        "probabilities": {c: round(float(p) * 100, 2) for c, p in zip(CLASSES, proba)},
    }


app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenAI Impact Predictor</title>
<style>
:root{--bg:#0f1226;--card:#181c3a;--line:#2a2f5c;--txt:#e8eaff;--mut:#9aa0d0;--a:#7c5cff;--b:#22d3ee;
--High:#34d399;--Medium:#fbbf24;--Low:#f87171}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Segoe UI,sans-serif;color:var(--txt);
background:radial-gradient(900px 500px at 10% -10%,#2b2470 0,transparent 60%),radial-gradient(800px 500px at 100% 0,#0b4a63 0,transparent 55%),var(--bg);min-height:100vh}
.wrap{max-width:980px;margin:0 auto;padding:40px 20px}
h1{font-size:2rem;margin:0 0 6px;background:linear-gradient(90deg,var(--a),var(--b));-webkit-background-clip:text;color:transparent}
p.sub{color:var(--mut);margin:0 0 28px}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:22px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:rgba(24,28,58,.85);border:1px solid var(--line);border-radius:18px;padding:24px;backdrop-filter:blur(8px)}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:560px){.fields{grid-template-columns:1fr}}
label{display:block;font-size:.78rem;color:var(--mut);margin-bottom:6px;letter-spacing:.03em}
input,select{width:100%;padding:11px 12px;border-radius:10px;border:1px solid var(--line);background:#10132b;color:var(--txt);font-size:.95rem}
input:focus,select:focus{outline:2px solid var(--a);border-color:transparent}
button{margin-top:20px;width:100%;padding:14px;border:0;border-radius:12px;font-size:1rem;font-weight:600;color:#fff;cursor:pointer;
background:linear-gradient(90deg,var(--a),var(--b));transition:transform .15s,opacity .15s}
button:hover{transform:translateY(-1px)}button:disabled{opacity:.6}
.res{display:flex;flex-direction:column;gap:16px;align-items:center;text-align:center}
.badge{font-size:2.4rem;font-weight:800;padding:14px 28px;border-radius:16px;border:2px solid var(--line)}
.muted{color:var(--mut);font-size:.9rem}
.bar{width:100%;text-align:left}.bar .row{display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:5px}
.track{height:10px;background:#10132b;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;border-radius:99px;transition:width .8s cubic-bezier(.2,.8,.2,1)}
.err{color:#fca5a5;font-size:.9rem}
</style></head><body><div class="wrap">
<h1>GenAI Impact Predictor</h1>
<p class="sub">Enter a student profile to predict the outcome class (High / Medium / Low).</p>
<div class="grid">
<form id="f" class="card"><div class="fields">
{% for name in features %}
<div><label>{{ labels[name] }}</label>
{% if name in cat %}<select name="{{ name }}">{% for o in cat[name] %}<option>{{ o }}</option>{% endfor %}</select>
{% else %}{% set n = num[name] %}<input type="number" name="{{ name }}" min="{{ n[1] }}" max="{{ n[2] }}" step="{{ n[3] }}" value="{{ n[4] }}" required>{% endif %}
</div>{% endfor %}
</div><button id="btn" type="submit">Predict</button></form>
<div class="card res" id="out"><div class="muted">Your prediction will appear here.</div></div>
</div></div>
<script>
const colors={High:'var(--High)',Medium:'var(--Medium)',Low:'var(--Low)'};
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const btn=document.getElementById('btn'),out=document.getElementById('out');
 btn.disabled=true;btn.textContent='Predicting…';
 const body=Object.fromEntries(new FormData(e.target).entries());
 try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();if(!r.ok)throw new Error(d.error||'Error');
  const c=colors[d.prediction]||'var(--a)';
  out.innerHTML=`<div class="muted">Predicted class</div><div class="badge" style="color:${c};border-color:${c}">${d.prediction}</div>
  <div class="muted">Confidence ${d.confidence}%</div><div class="bar" style="width:100%">`+
  Object.entries(d.probabilities).map(([k,v])=>`<div style="margin-bottom:12px"><div class="row"><span>${k}</span><span>${v}%</span></div>
  <div class="track"><div class="fill" data-w="${v}" style="background:${colors[k]||'var(--a)'}"></div></div></div>`).join('')+`</div>`;
  requestAnimationFrame(()=>document.querySelectorAll('.fill').forEach(x=>x.style.width=x.dataset.w+'%'));
 }catch(err){out.innerHTML=`<div class="err">${err.message}</div>`}
 btn.disabled=false;btn.textContent='Predict';});
</script></body></html>"""


@app.get("/")
def index():
    labels = {n: NUMERIC[n][0] if n in NUMERIC else n.replace("_", " ") for n in FEATURES}
    return render_template_string(PAGE, features=FEATURES, cat=CATEGORICAL, num=NUMERIC, labels=labels)


@app.post("/predict")
def predict_route():
    try:
        return jsonify(predict(request.get_json(force=True)))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # pragma: no cover
        return jsonify(error=f"Prediction failed: {e}"), 500


@app.get("/health")
def health():
    return jsonify(status="ok", classes=CLASSES, n_features=len(FEATURES))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
