
import streamlit as st
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from PIL import Image
import json, os, requests
from datetime import datetime

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0e1117,#1a1f2e)}
.title-box{background:linear-gradient(90deg,#1a6faf,#0d4f8c);padding:25px;border-radius:15px;text-align:center;margin-bottom:20px}
.result-tumor{background:linear-gradient(135deg,#ff416c,#ff4b2b);padding:20px;border-radius:15px;text-align:center}
.result-normal{background:linear-gradient(135deg,#11998e,#38ef7d);padding:20px;border-radius:15px;text-align:center}
.model-card{background:#1e2130;padding:12px;border-radius:10px;text-align:center;border:1px solid #444;margin:5px}
.report-box{background:#1e2130;padding:15px;border-radius:10px;border:1px solid #1a6faf}
</style>""", unsafe_allow_html=True)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
NUM_CLASSES = 4

MODELS_INFO = {
    'resnet_4class.pth': '1s33bLIVRlQvcWLoF-kE2HPOOREkquJJW',
    'vgg_4class.pth':    '1dCgXmAgp3h3wClV6_MgPy86Wx-wLh3W8',
    'eff_4class.pth':    '15hFh0WDQEaNHE_wtAkWZvAtfFzfCRfP5',
}

def download_model(file_id, dest):
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    r = session.get(url, params={'id': file_id}, stream=True)
    token = next((v for k,v in r.cookies.items() if k.startswith('download_warning')), None)
    if token:
        r = session.get(url, params={'id': file_id, 'confirm': token}, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(32768):
            if chunk: f.write(chunk)

def download_all():
    for fname, fid in MODELS_INFO.items():
        if not os.path.exists(fname) or os.path.getsize(fname) < 1000:
            st.info(f"⏳ Downloading {fname}...")
            download_model(fid, fname)
            st.success(f"✅ {fname} ready!")

download_all()

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, NUM_CLASSES))
    resnet.load_state_dict(torch.load('resnet_4class.pth', map_location=device))
    resnet.eval()
    vgg = models.vgg16(weights=None)
    vgg.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(4096, NUM_CLASSES))
    vgg.load_state_dict(torch.load('vgg_4class.pth', map_location=device))
    vgg.eval()
    eff = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
    eff.load_state_dict(torch.load('eff_4class.pth', map_location=device))
    eff.eval()
    return resnet, vgg, eff

resnet_m, vgg_m, eff_m = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(img_tensor):
    with torch.no_grad():
        r = torch.softmax(resnet_m(img_tensor.unsqueeze(0)), dim=1)[0]
        v = torch.softmax(vgg_m(img_tensor.unsqueeze(0)),    dim=1)[0]
        e = torch.softmax(eff_m(img_tensor.unsqueeze(0)),    dim=1)[0]
    avg = (r + v + e) / 3
    idx = avg.argmax().item()
    return {
        'result': CLASS_NAMES[idx], 'confidence': avg[idx].item()*100,
        'all_probs': avg.tolist(),
        'resnet': CLASS_NAMES[r.argmax().item()], 'resnet_conf': r.max().item()*100,
        'vgg':    CLASS_NAMES[v.argmax().item()], 'vgg_conf':    v.max().item()*100,
        'eff':    CLASS_NAMES[e.argmax().item()], 'eff_conf':    e.max().item()*100,
    }

def llama_report(tumor, conf, name, age, gender):
    try:
        from transformers import pipeline
        gen = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       torch_dtype=torch.float16, device_map="auto")
        desc = {'Glioma':'glial cell tumor','Meningioma':'benign brain membrane tumor',
                'Pituitary':'pituitary gland tumor','No Tumor':'no abnormal mass'}
        prompt = f"<|system|>You are a radiologist.<|user|>Write a brief MRI report for {name}, {age}y {gender}. Finding: {tumor} ({desc.get(tumor,'')}), Confidence {conf:.1f}%. Include Summary, Findings, Recommendation in 100 words.<|assistant|>"
        out = gen(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
        return out[0]['generated_text'].split('<|assistant|>')[-1].strip()
    except:
        return f"{tumor} detected with {conf:.1f}% confidence. Please consult a neurologist."

def load_hist():
    try:
        with open('history.json') as f: return json.load(f)
    except: return []

def save_hist(name, age, gender, result, conf):
    h = load_hist()
    h.append({'name':name,'age':age,'gender':gender,'result':result,
              'conf':f'{conf:.1f}%','date':datetime.now().strftime('%Y-%m-%d')})
    with open('history.json','w') as f: json.dump(h,f)

with st.sidebar:
    st.markdown("## 🧠 NeuroScan AI")
    st.markdown("---")
    patient_name = st.text_input("👤 Patient Name")
    age          = st.number_input("🎂 Age", 1, 120, 25)
    gender       = st.selectbox("Gender", ["Male","Female","Other"])
    st.markdown("---")
    st.markdown("### 📋 History")
    for r in load_hist()[-5:][::-1]:
        icon = "🔴" if r['result'] != "No Tumor" else "🟢"
        st.markdown(f"{icon} **{r['name']}** — {r['result']} ({r['date']})")

st.markdown('''<div class="title-box">
<h1 style="color:white;margin:0">🧠 NeuroScan AI</h1>
<p style="color:#ddd;margin:4px 0 0">Multi-class Tumor Detection | LLaMA Reports | 3D CNN</p>
</div>''', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded = st.file_uploader("Brain MRI image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded MRI", use_column_width=True)

with col2:
    if uploaded:
        st.markdown("### 🔍 Results")
        preds  = predict(transform(image))
        result = preds['result']
        conf   = preds['confidence']
        if result != "No Tumor":
            st.markdown(f'<div class="result-tumor"><h2 style="color:white">⚠️ {result.upper()} DETECTED</h2><h3 style="color:white">Confidence: {conf:.1f}%</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-normal"><h2 style="color:white">✅ NO TUMOR DETECTED</h2><h3 style="color:white">Confidence: {conf:.1f}%</h3></div>', unsafe_allow_html=True)

        st.markdown("#### 📊 All 4 Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(int(preds['all_probs'][i]*100), text=f"{cls}: {preds['all_probs'][i]*100:.1f}%")

        st.markdown("#### 🤖 Individual Model Results")
        m1,m2,m3 = st.columns(3)
        for col,nm,res,cn in [(m1,"ResNet18",preds['resnet'],preds['resnet_conf']),
                               (m2,"VGG16",   preds['vgg'],   preds['vgg_conf']),
                               (m3,"EffNet",  preds['eff'],   preds['eff_conf'])]:
            with col:
                st.markdown(f'<div class="model-card"><h5 style="color:#1a6faf">{nm}</h5><p style="color:white;margin:0">{res}</p><p style="color:#aaa;margin:0">{cn:.1f}%</p></div>', unsafe_allow_html=True)

        if patient_name:
            st.markdown("#### 📋 LLaMA Medical Report")
            with st.spinner("🤖 Generating report..."):
                report = llama_report(result, conf, patient_name, age, gender)
            st.markdown(f'<div class="report-box"><p style="color:#ddd">{report}</p></div>', unsafe_allow_html=True)
            save_hist(patient_name, age, gender, result, conf)
        else:
            st.info("👈 Enter patient name in sidebar for LLaMA report!")

print("✅ STEP 4 DONE — app.py ready!")
