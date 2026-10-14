import os
import glob
import time
import json
import joblib
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from concurrent.futures import ThreadPoolExecutor

os.makedirs("models", exist_ok=True)

RAGAS = [
    "Asavari", "Bageshree", "Bhairava", "Bhairavi", "Bhoopali",
    "Darbari", "Malkauns", "Sarang", "Yaman"
]

def extract_features_from_audio(file_path, duration=30):
    try:
        y, sr = sf.read(file_path)
        # Convert stereo to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Trim or pad to duration
        max_samples = int(duration * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
        elif len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)))
            
        y = y.astype(np.float32)
        
        # 1. 40 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Concatenate 40 + 40 + 40 = 120
        feature_matrix = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        feature_vector = np.mean(feature_matrix, axis=1)
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_single_file(item):
    path, label = item
    feat = extract_features_from_audio(path)
    return feat, label, path

def extract_dataset(folder="Datasets"):
    items = []
    for raga in os.listdir(folder):
        raga_dir = os.path.join(folder, raga)
        if os.path.isdir(raga_dir):
            for fname in os.listdir(raga_dir):
                if fname.lower().endswith(('.wav', '.mp3')):
                    items.append((os.path.join(raga_dir, fname), raga))
    
    print(f"Extracting features for {len(items)} files in {folder}...")
    X, y, paths = [], [], []
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_single_file, items))
        
    for feat, label, path in results:
        if feat is not None:
            X.append(feat)
            y.append(label)
            paths.append(path)
            
    return np.array(X), np.array(y), paths

def extract_test_dataset(folder="Test"):
    items = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.wav', '.mp3')):
            # Normalize label from filename (e.g. asavari01.wav -> Asavari, bhoop01 -> Bhoopali, dkanada -> Darbari)
            base = os.path.splitext(fname)[0]
            clean_name = ''.join([c for c in base if not c.isdigit()]).capitalize()
            if clean_name.lower().startswith('bhoop'):
                clean_name = 'Bhoopali'
            elif clean_name.lower().startswith('dkanada') or clean_name.lower().startswith('darbari'):
                clean_name = 'Darbari'
            elif clean_name.lower().startswith('malkauns'):
                clean_name = 'Malkauns'
            elif clean_name.lower().startswith('bageshree'):
                clean_name = 'Bageshree'
            elif clean_name.lower().startswith('bhairava'):
                clean_name = 'Bhairava'
            elif clean_name.lower().startswith('bhairavi'):
                clean_name = 'Bhairavi'
            elif clean_name.lower().startswith('asavari'):
                clean_name = 'Asavari'
            elif clean_name.lower().startswith('sarang'):
                clean_name = 'Sarang'
            elif clean_name.lower().startswith('yaman'):
                clean_name = 'Yaman'
                
            items.append((os.path.join(folder, fname), clean_name))
            
    print(f"Extracting features for {len(items)} files in {folder}...")
    X, y, paths = [], [], []
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_single_file, items))
        
    for feat, label, path in results:
        if feat is not None and label in RAGAS:
            X.append(feat)
            y.append(label)
            paths.append(path)
            
    return np.array(X), np.array(y), paths

# PyTorch Model Architecture
class RagaClassifierNet(nn.Module):
    def __init__(self, input_dim=120, num_classes=9):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.4)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x is (batch, 1, 120)
        feat = self.conv_block(x)
        out = self.fc(feat)
        return out

def main():
    start_time = time.time()
    
    # 1. Extract Training Data
    cache_path = "models/extracted_data.npz"
    if os.path.exists(cache_path):
        print("Loading cached extracted features...")
        data = np.load(cache_path, allow_pickle=True)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        X_train, y_train, _ = extract_dataset("Datasets")
        X_test, y_test, _ = extract_test_dataset("Test")
        np.savez(cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print("Features saved to cache.")

    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 2. Encode Labels & Standardize
    le = LabelEncoder()
    le.fit(RAGAS)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Also train an ensemble RandomForest & GradientBoosted classifier for high precision inference
    from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
    from xgboost import XGBClassifier
    
    xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    rf = ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42)
    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
    
    print("Training ML Ensemble...")
    ensemble.fit(X_train_scaled, y_train_enc)
    ens_test_preds = ensemble.predict(X_test_scaled)
    ens_acc = accuracy_score(y_test_enc, ens_test_preds)
    print(f"Ensemble Test Accuracy: {ens_acc * 100:.2f}%")
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    
    # 3. PyTorch Deep CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch CNN on device: {device}")
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train_enc, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test_enc, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = RagaClassifierNet(input_dim=120, num_classes=len(RAGAS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    best_acc = 0.0
    history = {"train_loss": [], "test_acc": []}
    
    for epoch in range(1, 61):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        test_acc = correct / total
        history["train_loss"].append(total_loss / len(train_loader))
        history["test_acc"].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/best_cnn_model.pt")
            
        if epoch % 10 == 0 or epoch == 60:
            print(f"Epoch [{epoch}/60] - Loss: {total_loss/len(train_loader):.4f} - Test Acc: {test_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")
            
    # Load best model for final report
    model.load_state_dict(torch.load("models/best_cnn_model.pt"))
    model.eval()
    
    with torch.no_grad():
        all_preds = []
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    final_acc = accuracy_score(y_test_enc, all_preds)
    cm = confusion_matrix(y_test_enc, all_preds).tolist()
    report = classification_report(y_test_enc, all_preds, target_names=RAGAS, output_dict=True)
    
    metrics = {
        "cnn_test_accuracy": round(final_acc * 100, 2),
        "ensemble_test_accuracy": round(ens_acc * 100, 2),
        "best_accuracy": round(max(final_acc, ens_acc) * 100, 2),
        "confusion_matrix": cm,
        "classes": RAGAS,
        "classification_report": report,
        "training_history": history,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test)
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\n================ Training Complete in {time.time()-start_time:.1f}s ================")
    print(f"CNN Accuracy: {final_acc * 100:.2f}% | Ensemble Accuracy: {ens_acc * 100:.2f}%")
    print(classification_report(y_test_enc, all_preds, target_names=RAGAS))

if __name__ == "__main__":
    main()
