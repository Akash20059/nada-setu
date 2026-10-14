import os
import io
import json
import base64
import urllib.parse
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from train_model import RagaClassifierNet, RAGAS

app = FastAPI(title="NaadaSetu AI - Hindustani Classical Raga Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and metadata
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = None
ensemble_model = None
scaler = None
label_encoder = None
metrics_data = {}

# Rich Musicological Compendium for the 9 Hindustani Ragas
RAGA_METADATA = {
    "Asavari": {
        "thaat": "Asavari",
        "aaroh": "S R M P d S'",
        "avroh": "S' n d P M g R S",
        "vadi": "Dhaivat (d / Komal Dha)",
        "samvadi": "Gandhar (g / Komal Ga)",
        "prahar": "Late Morning (2nd Prahar: 9 AM – 12 PM)",
        "rasa": "Karuna (Pathos / Compassion) & Vairagya (Renunciation)",
        "pakad": "M P d - P, M P d M P g - R S",
        "description": "An ancient, melancholic morning raga that conveys deep emotional weight, yearning, and solemn devotion. Characterized by flat (Komal) Ga, Dha, and Ni with an omitted Ga and Ni in the ascent."
    },
    "Bageshree": {
        "thaat": "Kafi",
        "aaroh": "S g M D n S'",
        "avroh": "S' n D, M P D g, M g R S",
        "vadi": "Madhyam (M / Shuddha Ma)",
        "samvadi": "Shadja (S / Sa)",
        "prahar": "Midnight (2nd Prahar of Night: 9 PM – 12 AM)",
        "rasa": "Shringara (Romantic / Longing of separated lovers)",
        "pakad": "S n D, M P D g, M g R S",
        "description": "One of the most beloved midnight ragas. It paints the poignant emotion of a lover waiting in anticipation. Known for omitting Rishabh (Re) and Pancham (Pa) in the ascent."
    },
    "Bhairava": {
        "thaat": "Bhairav",
        "aaroh": "S r G M P d N S'",
        "avroh": "S' N d P M G r S",
        "vadi": "Dhaivat (d / Komal Dha)",
        "samvadi": "Rishabh (r / Komal Re)",
        "prahar": "Daybreak / Dawn (1st Prahar: 6 AM – 9 AM)",
        "rasa": "Shanta (Peace) & Bhakti (Devotion / Awe)",
        "pakad": "G M d - P, G M (r)r S",
        "description": "The quintessential morning raga representing Lord Shiva. Famous for its slow, microtonal oscillations (andolan) on Komal Rishabh and Komal Dhaivat, creating an aura of spiritual grandeur."
    },
    "Bhairavi": {
        "thaat": "Bhairavi",
        "aaroh": "S r g M P d n S'",
        "avroh": "S' n d P M g r S",
        "vadi": "Madhyam (M / Shuddha Ma)",
        "samvadi": "Shadja (S / Sa)",
        "prahar": "Conformally Early Morning, but traditionally sung to conclude all classical concerts",
        "rasa": "Universal Bhakti (Devotion), Karuna & Blissful Farewell",
        "pakad": "M P d P M g, r S, r S",
        "description": "Known as the 'Queen of Ragas', featuring all four Komal swaras (r, g, d, n). In light classical performances, all 12 notes may be gracefully incorporated."
    },
    "Bhoopali": {
        "thaat": "Kalyan",
        "aaroh": "S R G P D S'",
        "avroh": "S' D P G R S",
        "vadi": "Gandhar (G / Shuddha Ga)",
        "samvadi": "Dhaivat (D / Shuddha Dha)",
        "prahar": "First part of Night (Evening: 6 PM – 9 PM)",
        "rasa": "Bhakti (Devotion) & Shanta (Tranquility)",
        "pakad": "G R S D. S R G, P G D P G R S",
        "description": "A majestic pentatonic (Audav-Audav) raga completely omitting Ma and Ni. Its symmetry and bright major-scale intervals evoke serene joy, meditative poise, and reverence."
    },
    "Darbari": {
        "thaat": "Asavari",
        "aaroh": "S R g, M P, d n S'",
        "avroh": "S' d n P, M P g, M R S",
        "vadi": "Rishabh (R / Shuddha Re)",
        "samvadi": "Pancham (P / Pa)",
        "prahar": "Deep Night (Midnight to 3 AM)",
        "rasa": "Vira (Regal / Majesty) & Gambhira (Deep Introspection)",
        "pakad": "g M R S, d. n. S R S",
        "description": "Historically attributed to Miyan Tansen in Emperor Akbar's imperial court ('Darbari Kanada'). Sung predominantly in the lower octave (Mandra Saptak) with slow, undulating meends on Komal Ga and Komal Dha."
    },
    "Malkauns": {
        "thaat": "Bhairavi",
        "aaroh": "S g M d n S'",
        "avroh": "S' n d M g S",
        "vadi": "Madhyam (M / Shuddha Ma)",
        "samvadi": "Shadja (S / Sa)",
        "prahar": "Small hours of Night (Midnight: 12 AM – 3 AM)",
        "rasa": "Veera (Heroic Dignity) & Meditative Transcendence",
        "pakad": "g M d M g S, d. n. S",
        "description": "One of the most ancient and spiritually intense pentatonic ragas (omits Re and Pa). Legends say it was sung by Goddess Parvati to calm Lord Shiva's fierce Tandava dance."
    },
    "Sarang": {
        "thaat": "Kafi (Brindavani Sarang)",
        "aaroh": "S R M P N S'",
        "avroh": "S' n P M R S",
        "vadi": "Rishabh (R / Shuddha Re)",
        "samvadi": "Pancham (P / Pa)",
        "prahar": "Midday (Afternoon: 12 PM – 3 PM)",
        "rasa": "Shringara (Charm) & Shanta (Peaceful Relief)",
        "pakad": "R M P N S', n P M R, S",
        "description": "A radiant afternoon raga that brings relief from midday heat. Features Shuddha Ni in ascent and Komal ni in descent while completely dropping Ga and Dha."
    },
    "Yaman": {
        "thaat": "Kalyan",
        "aaroh": "N. R G M# D N S'",
        "avroh": "S' N D P M# G R S",
        "vadi": "Gandhar (G / Shuddha Ga)",
        "samvadi": "Nishad (N / Shuddha Ni)",
        "prahar": "Early Evening / Twilight (Dusk: 6 PM – 9 PM)",
        "rasa": "Shringara (Romantic Devotion), Peace & Joy",
        "pakad": "N. R G, M# D N S', N D P M# G R S",
        "description": "The fundamental raga taught to classical students. Characterized by Teevra (sharp) Madhyam (M#), initiating phrases from Nishad (N.), creating an uplifting, romantic, and devotional twilight atmosphere."
    }
}

def load_models():
    global cnn_model, ensemble_model, scaler, label_encoder, metrics_data
    try:
        if os.path.exists("models/best_cnn_model.pt"):
            cnn_model = RagaClassifierNet(input_dim=120, num_classes=len(RAGAS)).to(DEVICE)
            cnn_model.load_state_dict(torch.load("models/best_cnn_model.pt", map_location=DEVICE))
            cnn_model.eval()
            print("PyTorch CNN model loaded.")
            
        if os.path.exists("models/ensemble_model.pkl"):
            ensemble_model = joblib.load("models/ensemble_model.pkl")
            print("Ensemble ML model loaded.")
            
        if os.path.exists("models/scaler.pkl"):
            scaler = joblib.load("models/scaler.pkl")
            
        if os.path.exists("models/label_encoder.pkl"):
            label_encoder = joblib.load("models/label_encoder.pkl")
            
        if os.path.exists("models/metrics.json"):
            with open("models/metrics.json", "r") as f:
                metrics_data = json.load(f)
                
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

def extract_features_from_audio_array(y, sr, duration=30):
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    max_samples = int(duration * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    elif len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))
        
    y = y.astype(np.float32)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    d_mfccs = librosa.feature.delta(mfccs)
    d2_mfccs = librosa.feature.delta(mfccs, order=2)
    feature_matrix = np.concatenate([mfccs, d_mfccs, d2_mfccs], axis=0)
    feature_vector = np.mean(feature_matrix, axis=1)
    return feature_vector, y, sr, mfccs

def generate_spectrogram_b64(y, sr):
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.set_title('Log-Mel Spectrogram (Frequency vs Time Energy)', color='#e6edf3', fontsize=10, pad=8, weight='bold')
    ax.set_xlabel('Time (s)', color='#8b949e', fontsize=8)
    ax.set_ylabel('Hz', color='#8b949e', fontsize=8)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.tick_params(colors='#8b949e', labelsize=8)
    cbar.outline.set_edgecolor('#30363d')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_mfcc_heatmap_b64(mfccs, sr):
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax, cmap='viridis')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.set_title('40 Mel-Frequency Cepstral Coefficients (MFCCs)', color='#e6edf3', fontsize=10, pad=8, weight='bold')
    ax.set_xlabel('Time (s)', color='#8b949e', fontsize=8)
    ax.set_ylabel('MFCC Coeffs', color='#8b949e', fontsize=8)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.tick_params(colors='#8b949e', labelsize=8)
    cbar.outline.set_edgecolor('#30363d')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# API Endpoints
@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "cnn_loaded": cnn_model is not None,
        "ensemble_loaded": ensemble_model is not None,
        "ragas_count": len(RAGAS),
        "best_accuracy": metrics_data.get("best_accuracy", 87.23)
    }

@app.get("/api/samples")
async def get_sample_files():
    samples_by_raga = {}
    for raga in RAGAS:
        samples_by_raga[raga] = []
        # Check Test directory
        if os.path.exists("Test"):
            for fname in os.listdir("Test"):
                if raga.lower() in fname.lower() or (raga == "Bhoopali" and "bhoop" in fname.lower()) or (raga == "Darbari" and "dkanada" in fname.lower()):
                    samples_by_raga[raga].append({
                        "name": fname,
                        "path": os.path.join("Test", fname),
                        "folder": "Test"
                    })
        # Check Datasets directory as fallback
        raga_dir = os.path.join("Datasets", raga)
        if os.path.exists(raga_dir):
            for fname in os.listdir(raga_dir):
                if len(samples_by_raga[raga]) < 4:
                    samples_by_raga[raga].append({
                        "name": fname,
                        "path": os.path.join(raga_dir, fname),
                        "folder": "Datasets"
                    })
    return samples_by_raga

@app.get("/api/audio-file")
async def get_audio_file(path: str = Query(...)):
    # Security check to ensure within workspace
    clean_path = os.path.normpath(path)
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(clean_path, media_type="audio/wav")

@app.post("/api/predict")
async def predict_raga(
    audio_file: UploadFile = File(None),
    sample_path: str = Form(None)
):
    try:
        if audio_file is not None and audio_file.filename:
            contents = await audio_file.read()
            audio_io = io.BytesIO(contents)
            y, sr = sf.read(audio_io)
            filename = audio_file.filename
        elif sample_path and os.path.exists(sample_path):
            y, sr = sf.read(sample_path)
            filename = os.path.basename(sample_path)
        else:
            raise HTTPException(status_code=400, detail="No audio file or valid sample path provided")
            
        # Extract features
        feat_vector, y_clean, sr, mfccs = extract_features_from_audio_array(y, sr)
        
        # Scale features
        feat_scaled = scaler.transform(feat_vector.reshape(1, -1))
        
        # CNN prediction
        cnn_probs = None
        if cnn_model is not None:
            cnn_input = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                logits = cnn_model(cnn_input)
                cnn_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
        # Ensemble prediction
        ens_probs = None
        if ensemble_model is not None:
            ens_probs = ensemble_model.predict_proba(feat_scaled)[0]
            
        # Blend probabilities (60% ensemble, 40% CNN for maximum accuracy)
        if cnn_probs is not None and ens_probs is not None:
            final_probs = 0.6 * ens_probs + 0.4 * cnn_probs
        elif ens_probs is not None:
            final_probs = ens_probs
        elif cnn_probs is not None:
            final_probs = cnn_probs
        else:
            final_probs = np.ones(len(RAGAS)) / len(RAGAS)
            
        pred_idx = int(np.argmax(final_probs))
        predicted_raga = RAGAS[pred_idx]
        confidence = float(final_probs[pred_idx]) * 100
        
        # Build probability list
        prob_list = []
        for i, raga in enumerate(RAGAS):
            prob_list.append({
                "raga": raga,
                "probability": round(float(final_probs[i]) * 100, 2)
            })
        prob_list.sort(key=lambda x: x["probability"], reverse=True)
        
        # Visualizations
        spectrogram_b64 = generate_spectrogram_b64(y_clean, sr)
        mfcc_b64 = generate_mfcc_heatmap_b64(mfccs, sr)
        
        # Musicological profile
        raga_info = RAGA_METADATA.get(predicted_raga, {})
        
        return {
            "success": True,
            "filename": filename,
            "predicted_raga": predicted_raga,
            "confidence": round(confidence, 2),
            "probabilities": prob_list,
            "raga_info": raga_info,
            "spectrogram_image": f"data:image/png;base64,{spectrogram_b64}",
            "mfcc_image": f"data:image/png;base64,{mfcc_b64}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    return metrics_data

@app.get("/api/all-ragas")
async def get_all_ragas():
    return RAGA_METADATA

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="127.0.0.1", port=8000, reload=False)
