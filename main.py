from pathlib import Path

import cv2
import joblib
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "rf_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="🍌 Banana Ripeness Classifier API",
    description="AI-powered banana ripeness classification",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SELF-CONTAINED FRONTEND
# ============================================================

APP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BananaIQ · Ripeness Intelligence</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#080d13;--panel:#101923;--panel2:#151f2b;--line:#263442;--text:#f4f7fb;--muted:#8e9baa;--lime:#c8f36b;--cyan:#75e3d1;--orange:#ffb86b;--red:#ff7f91;--shadow:0 24px 70px #0007}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(900px 500px at 85% -10%,#24483e55,transparent 60%),radial-gradient(700px 500px at -10% 20%,#30405b44,transparent 60%),var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}button,input{font:inherit}.shell{width:min(1180px,calc(100% - 36px));margin:auto;padding:28px 0 42px}.top{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:46px}.brand{display:flex;gap:12px;align-items:center}.mark{width:46px;height:46px;border-radius:15px;display:grid;place-items:center;background:var(--lime);color:#18220d;font-size:25px;box-shadow:0 0 0 7px #c8f36b12}.eyebrow{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--lime);font-weight:700}.brand h1{font:700 22px 'Space Grotesk';margin:3px 0}.brand p{margin:0;color:var(--muted);font-size:12px}.nav{display:flex;gap:9px;flex-wrap:wrap}.nav a,.ghost{border:1px solid var(--line);border-radius:11px;padding:10px 13px;color:var(--text);text-decoration:none;background:#ffffff05;font-size:13px}.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:28px;align-items:end;margin-bottom:30px}.hero h2{font:700 clamp(34px,5vw,65px)/.98 'Space Grotesk';letter-spacing:-3px;margin:10px 0 16px;max-width:720px}.hero h2 span{color:var(--lime)}.hero p{color:var(--muted);line-height:1.7;max-width:610px;margin:0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.stat{padding:17px;border:1px solid var(--line);border-radius:16px;background:#ffffff04}.stat b{display:block;font:700 25px 'Space Grotesk';margin-bottom:5px}.stat small{color:var(--muted);font-size:11px}.layout{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{border:1px solid var(--line);border-radius:24px;background:linear-gradient(145deg,#14202c,#0d151e);box-shadow:var(--shadow);padding:25px}.cardhead{display:flex;justify-content:space-between;align-items:start;gap:15px;margin-bottom:20px}.card h3{font:600 20px 'Space Grotesk';margin:0 0 6px}.muted{color:var(--muted);font-size:13px;line-height:1.6}.tag{border:1px solid #c8f36b44;color:var(--lime);background:#c8f36b0d;border-radius:999px;padding:6px 9px;font-size:10px;font-weight:700;white-space:nowrap}.drop{min-height:320px;border:1px dashed #526779;border-radius:19px;display:flex;align-items:center;justify-content:center;text-align:center;padding:25px;transition:.2s;background:linear-gradient(135deg,#ffffff03,#c8f36b03)}.drop.active{border-color:var(--lime);background:#c8f36b0b}.dropinner .uploadicon{width:70px;height:70px;border:1px solid #526779;border-radius:22px;display:grid;place-items:center;font-size:30px;margin:0 auto 17px;background:#ffffff05}.drop strong{display:block;font-size:16px;margin-bottom:8px}.drop small{color:var(--muted)}input[type=file]{display:none}.choose{display:inline-block;margin-top:20px;background:var(--lime);color:#18220d;padding:12px 17px;border-radius:12px;font-weight:700;font-size:13px;cursor:pointer}.filename{margin-top:14px;color:var(--cyan);font-size:12px;overflow-wrap:anywhere}.preview{display:none;width:100%;height:320px;object-fit:contain;border-radius:19px;background:#080d13;margin-bottom:14px;border:1px solid var(--line)}.actions{display:flex;gap:10px;margin-top:15px}.btn{border:1px solid var(--line);border-radius:12px;padding:13px 15px;cursor:pointer;color:var(--text);background:#ffffff05;font-weight:600;font-size:13px}.primary{flex:1;background:var(--lime);border-color:var(--lime);color:#17210d}.btn:disabled{opacity:.35;cursor:not-allowed}.status{min-height:20px;color:var(--muted);font-size:12px;margin-top:13px}.empty{min-height:320px;border:1px dashed var(--line);border-radius:19px;display:flex;align-items:center;justify-content:center;text-align:center;padding:35px;color:var(--muted);line-height:1.7}.result{display:none}.resulttop{display:flex;justify-content:space-between;align-items:center;gap:10px}.resultlabel{font:700 clamp(30px,4vw,48px) 'Space Grotesk';letter-spacing:-1.5px;text-transform:capitalize;margin:24px 0 5px}.confidence{font-size:13px;color:var(--muted);margin-bottom:24px}.confidence strong{color:var(--lime);font-size:18px}.bar{margin:17px 0}.bartop{display:flex;justify-content:space-between;font-size:12px;margin-bottom:8px;text-transform:capitalize}.track{height:9px;background:#263442;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--lime));transition:width .7s ease}.footer{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;margin-top:22px;color:var(--muted);font-size:11px}.notice{margin-top:18px;border:1px solid #75e3d133;background:#75e3d108;padding:13px 15px;border-radius:14px;font-size:12px;color:#b6c4cf;line-height:1.6}.spin{display:inline-block;width:13px;height:13px;border:2px solid #18220d55;border-top-color:#18220d;border-radius:50%;animation:spin .7s linear infinite;vertical-align:-2px;margin-right:6px}@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:900px){.hero,.layout{grid-template-columns:1fr}.hero h2{letter-spacing:-2px}.stats{max-width:600px}}@media(max-width:560px){.shell{width:min(100% - 24px,1180px);padding-top:18px}.top{align-items:flex-start;flex-direction:column;margin-bottom:32px}.card{padding:18px;border-radius:20px}.drop,.empty{min-height:270px}.preview{height:270px}.stats{gap:7px}.stat{padding:12px}.stat b{font-size:20px}}
</style>
</head>
<body>
<div class="shell">
<header class="top"><div class="brand"><div class="mark">🍌</div><div><div class="eyebrow">Computer Vision Lab</div><h1>BananaIQ</h1><p>Ripeness intelligence, made simple.</p></div></div><nav class="nav"><a href="/docs" target="_blank">API Docs ↗</a><a href="/health" target="_blank">System Health ↗</a></nav></header>
<section class="hero"><div><div class="eyebrow">AI-powered classification</div><h2>Know your banana.<br><span>Before it goes bad.</span></h2><p>Upload a banana image and let the trained machine-learning pipeline estimate its ripeness stage using handcrafted computer-vision features.</p></div><div class="stats"><div class="stat"><b>4</b><small>Ripeness classes</small></div><div class="stat"><b>103</b><small>Image features</small></div><div class="stat"><b>91.01%</b><small>Test accuracy</small></div></div></section>
<section class="layout"><article class="card"><div class="cardhead"><div><h3>Inspect an image</h3><div class="muted">JPG, PNG or BMP · Clear single-banana images work best.</div></div><span class="tag">UPLOAD</span></div><div class="drop" id="drop"><div class="dropinner"><div class="uploadicon">↥</div><strong>Drop your image here</strong><small>or select a file from your device</small><label class="choose" for="file">Choose image</label><input id="file" type="file" accept="image/jpeg,image/png,image/bmp"><div class="filename" id="filename">No image selected</div></div></div><img id="preview" class="preview" alt="Selected banana image"><div class="actions"><button class="btn primary" id="predict" disabled>Analyze image →</button><button class="btn" id="clear">Reset</button></div><div class="status" id="status" role="status"></div><div class="notice">Your image is processed by the local FastAPI service. The result is an estimate, not a substitute for human inspection.</div></article>
<article class="card"><div class="cardhead"><div><h3>Classification report</h3><div class="muted">Prediction confidence and class probabilities.</div></div><span class="tag">MODEL OUTPUT</span></div><div class="empty" id="empty">Your prediction will appear here.<br>Upload an image to begin analysis.</div><div class="result" id="result"><div class="resulttop"><span class="tag">ANALYSIS COMPLETE</span><span class="muted" id="resultfile">—</span></div><div class="resultlabel" id="label">—</div><div class="confidence">Model confidence: <strong id="confidence">—</strong></div><div id="bars"></div></div></article></section>
<footer class="footer"><span>Extra Trees classifier · OpenCV · Scikit-learn</span><span>FastAPI backend · BananaIQ</span></footer>
</div>
<script>
const $=id=>document.getElementById(id);const fileInput=$('file'),drop=$('drop'),preview=$('preview'),filename=$('filename'),predict=$('predict'),clear=$('clear'),status=$('status'),empty=$('empty'),result=$('result');let selected=null;
function choose(file){if(!file)return;const allowed=['image/jpeg','image/png','image/bmp'];if(!allowed.includes(file.type)){status.textContent='Please select a JPG, PNG, or BMP image.';return}selected=file;filename.textContent=file.name;preview.src=URL.createObjectURL(file);preview.style.display='block';drop.style.display='none';predict.disabled=false;status.textContent='Image ready for analysis.';empty.style.display='flex';result.style.display='none'}
fileInput.addEventListener('change',e=>choose(e.target.files[0]));['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('active')}));['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('active')}));drop.addEventListener('drop',e=>choose(e.dataTransfer.files[0]));
predict.addEventListener('click',async()=>{if(!selected)return;const data=new FormData();data.append('file',selected);predict.disabled=true;predict.innerHTML='<span class="spin"></span>Analyzing';status.textContent='Extracting features and running prediction…';try{const response=await fetch('/predict',{method:'POST',body:data});const payload=await response.json();if(!response.ok)throw new Error(payload.detail||'Prediction failed.');$('label').textContent=payload.predicted_class;$('confidence').textContent=Number(payload.confidence).toFixed(2)+'%';$('resultfile').textContent=selected.name;$('bars').innerHTML=Object.entries(payload.probabilities).sort((a,b)=>b[1]-a[1]).map(([name,value])=>'<div class="bar"><div class="bartop"><span>'+name+'</span><b>'+Number(value).toFixed(2)+'%</b></div><div class="track"><div class="fill" style="width:'+Math.min(100,Math.max(0,Number(value)))+'%"></div></div></div>').join('');empty.style.display='none';result.style.display='block';status.textContent='Prediction completed successfully.'}catch(err){status.textContent=err.message}finally{predict.disabled=false;predict.textContent='Analyze image →'}});
clear.addEventListener('click',()=>{selected=null;fileInput.value='';filename.textContent='No image selected';preview.removeAttribute('src');preview.style.display='none';drop.style.display='flex';predict.disabled=true;status.textContent='';empty.style.display='flex';result.style.display='none';predict.textContent='Analyze image →'});
</script></body></html>
"""


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found: {MODEL_PATH}\n"
        "Run 03_TrainModel.py first."
    )

if not ENCODER_PATH.exists():
    raise RuntimeError(
        f"Label encoder not found: {ENCODER_PATH}\n"
        "Run 03_TrainModel.py first."
    )


model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image: np.ndarray) -> np.ndarray:

    # Resize
    image = cv2.resize(image, (224, 224))

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    blur = cv2.GaussianBlur(
        hsv,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # COLOR RANGES
    # --------------------------------------------------------

    lower_green = np.array([
        35,
        40,
        40
    ])

    upper_green = np.array([
        85,
        255,
        255
    ])

    lower_yellow = np.array([
        20,
        100,
        100
    ])

    upper_yellow = np.array([
        35,
        255,
        255
    ])

    lower_brown = np.array([
        5,
        50,
        20
    ])

    upper_brown = np.array([
        20,
        255,
        200
    ])

    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    mask_green = cv2.inRange(
        blur,
        lower_green,
        upper_green
    )

    mask_yellow = cv2.inRange(
        blur,
        lower_yellow,
        upper_yellow
    )

    mask_brown = cv2.inRange(
        blur,
        lower_brown,
        upper_brown
    )

    mask = (
        mask_green
        | mask_yellow
        | mask_brown
    )

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    banana_mask = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            banana_mask,
            [largest],
            -1,
            255,
            -1
        )

    else:

        banana_mask = mask

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=banana_mask
    )

    # --------------------------------------------------------
    # MEAN
    # --------------------------------------------------------

    mean = cv2.mean(
        segmented,
        mask=banana_mask
    )[:3]

    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    std = cv2.meanStdDev(
        segmented,
        mask=banana_mask
    )[1].flatten()

    # --------------------------------------------------------
    # BGR HISTOGRAMS
    # --------------------------------------------------------

    hist_b = cv2.calcHist(
        [segmented],
        [0],
        banana_mask,
        [16],
        [0, 256]
    )

    hist_g = cv2.calcHist(
        [segmented],
        [1],
        banana_mask,
        [16],
        [0, 256]
    )

    hist_r = cv2.calcHist(
        [segmented],
        [2],
        banana_mask,
        [16],
        [0, 256]
    )

    # --------------------------------------------------------
    # HSV HISTOGRAMS
    # --------------------------------------------------------

    hsv_image = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2HSV
    )

    hist_h = cv2.calcHist(
        [hsv_image],
        [0],
        banana_mask,
        [16],
        [0, 180]
    )

    hist_s = cv2.calcHist(
        [hsv_image],
        [1],
        banana_mask,
        [16],
        [0, 256]
    )

    hist_v = cv2.calcHist(
        [hsv_image],
        [2],
        banana_mask,
        [16],
        [0, 256]
    )

    # --------------------------------------------------------
    # NORMALIZE HISTOGRAMS
    # --------------------------------------------------------

    hist_b = cv2.normalize(
        hist_b,
        hist_b
    ).flatten()

    hist_g = cv2.normalize(
        hist_g,
        hist_g
    ).flatten()

    hist_r = cv2.normalize(
        hist_r,
        hist_r
    ).flatten()

    hist_h = cv2.normalize(
        hist_h,
        hist_h
    ).flatten()

    hist_s = cv2.normalize(
        hist_s,
        hist_s
    ).flatten()

    hist_v = cv2.normalize(
        hist_v,
        hist_v
    ).flatten()

    # --------------------------------------------------------
    # EDGE DENSITY
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = (
        np.sum(edges > 0)
        / (224 * 224)
    )

    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    feature_vector = np.hstack([
        mean,
        std,

        hist_b,
        hist_g,
        hist_r,

        hist_h,
        hist_s,
        hist_v,

        edge_density
    ])

    return feature_vector.reshape(
        1,
        -1
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTMLResponse(content=APP_HTML)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "classes": encoder.classes_.tolist()
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE FILE TYPE
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/bmp"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Please upload JPG, PNG, or BMP."
            )
        )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # DECODE IMAGE
    # --------------------------------------------------------

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded image."
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        feature_vector = extract_features(
            image
        )

        prediction = model.predict(
            feature_vector
        )

        predicted_label = (
            encoder.inverse_transform(
                prediction
            )[0]
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            feature_vector
        )[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        class_probabilities = {}

        for label, probability in zip(
            encoder.classes_,
            probabilities
        ):

            class_probabilities[
                str(label)
            ] = round(
                float(probability * 100),
                2
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": str(
                predicted_label
            ),
            "confidence": round(
                confidence,
                2
            ),
            "probabilities": class_probabilities
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )