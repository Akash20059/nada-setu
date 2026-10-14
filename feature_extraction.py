import librosa
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

ragas = []
features = []
mel_features = []

class audio_augmentation:

    def time_stretch(self, audio_file, factor=None):
        if factor is None:
           factor = random.uniform(0.8, 1.2)
        y_stretched = librosa.effects.time_stretch(audio_file, rate=factor)
        return y_stretched
    
    def pitch_shift(self, audio_file, sr, factor=None):
        if factor is None:
            factor = random.uniform(-2, 2)
        y_shifted = librosa.effects.pitch_shift(audio_file, sr=sr, n_steps=factor)
        return y_shifted
    
    def noise_addition(self, audio_file):
        noise = 0.005 * np.random.randn(len(audio_file))
        y_noisy = audio_file + noise
        return y_noisy
  
def create_spectrogram(file, n_fft, hop):
    try:
        stft_output = librosa.stft(file, n_fft=n_fft, hop_length=hop)
        spectrogram = np.abs(stft_output)
        return spectrogram
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None
    
def extract_features_mel(file, sr, n_mels=128, max_pad_len=174):
    mel_spec = librosa.feature.melspectrogram(y=file, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    flattened_mel_spec = mel_spec_db.flatten()
    return flattened_mel_spec

def preprocess_dataset(feature_list):
    X = np.array(feature_list, dtype=object)
    max_length = max(len(row) for row in X)
    padded_X = np.zeros((len(X), max_length), dtype=np.float32)
    for i, row in enumerate(X):
        padded_X[i, :len(row)] = row
    df = pd.DataFrame(padded_X)
    return df

def plot_spectrogram(spectrogram, sr, hop_length):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                             sr=sr,
                             hop_length=hop_length,
                             x_axis='time',
                             y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()


#Feature extractor using mfcc
def extract_mfcc_feature_vector(audio_file, sr):
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    feature_matrix = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
    feature_vector = np.mean(feature_matrix, axis=1)
    return feature_vector

def create_mfcc_dataset():
    # Iterate through the folders and the files
    for ragas_folder in os.listdir("Datasets/"):
        ragas_path = os.path.join("Datasets/", ragas_folder)
        if os.path.isdir(ragas_path):
            for filename in os.listdir(ragas_path):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    audio_path = os.path.join(ragas_path, filename)
                try:
                    y, sr = librosa.load(audio_path, duration=30)
                    feature_vector = extract_mfcc_feature_vector(y, sr)
                    features.append(feature_vector)
                    ragas.append(ragas_folder)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    feature_df = pd.DataFrame(features)
    ragas_df = pd.DataFrame({'Ragas': ragas})
    dataset = pd.concat([feature_df, ragas_df], axis=1)
    prefix = "mfcc_"
    new_columns = [prefix + str(col) for col in dataset.columns[:-1]]
    dataset.columns = new_columns + [dataset.columns[-1]]
    return dataset

def create_mfcc_dataset_with_audio_aug():
    # Iterate through the folders and the files
    for ragas_folder in os.listdir("Datasets/"):
        ragas_path = os.path.join("Datasets/", ragas_folder)
        if os.path.isdir(ragas_path):
            for filename in os.listdir(ragas_path):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    audio_path = os.path.join(ragas_path, filename)
                try:
                    y, sr = librosa.load(audio_path, duration=30)
                    a = audio_augmentation()
                    y_stretched = a.time_stretch(y)
                    y_shifted = a.pitch_shift(y_stretched,sr)
                    #aug_audio = a.noise_addition(y_shifted)
                    feature_vector = extract_mfcc_feature_vector(y_shifted, sr)
                    features.append(feature_vector)
                    ragas.append(ragas_folder)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    feature_df = pd.DataFrame(features)
    ragas_df = pd.DataFrame({'Ragas': ragas})
    dataset = pd.concat([feature_df, ragas_df], axis=1)
    prefix = "mfcc_"
    new_columns = [prefix + str(col) for col in dataset.columns[:-1]]
    dataset.columns = new_columns + [dataset.columns[-1]]
    return dataset

def create_melSpectogram_dataset():
    for ragas_folder in os.listdir("Datasets/"):
        ragas_path = os.path.join("Datasets/", ragas_folder)
        if os.path.isdir(ragas_path):
           for filename in os.listdir(ragas_path):
               if filename.endswith(".wav") or filename.endswith(".mp3"):
                   audio_path = os.path.join(ragas_path, filename)
               try:
                   y, sr = librosa.load(audio_path, duration=30)
                   feature_vector = extract_features_mel(y, sr)
                   mel_features.append(feature_vector)
                   ragas.append(ragas_folder)
               except Exception as e:
                   print(f"Error processing {filename}: {e}")

    feature_df = preprocess_dataset(mel_features)
    ragas_df = pd.DataFrame({'Ragas': ragas})
    mel_dataset = pd.concat([feature_df, ragas_df], axis=1)
    prefix = "mel_"
    new_columns = [prefix + str(col) for col in mel_dataset.columns[:-1]]
    mel_dataset.columns = new_columns + [mel_dataset.columns[-1]]
    return mel_dataset

if __name__ == "__main__":
    print("Please select the dataset type.")
    while True:
        print("\n--- Select a dataset type to create ---")
        print("1. MFCC Feature")
        print("2. MFCC features with Audio Augmentation")
        print("3. Mel Spectogram")
        print("4. Exit")

        feature_selection = input("Enter your choice (1-4): ")

        if feature_selection == '1':
            dataset = create_mfcc_dataset()
            file_name = "mfcc_dataset.csv"
        elif feature_selection == '2':
            dataset = create_mfcc_dataset_with_audio_aug()
            file_name = "audio_aug_mfcc_dataset.csv"
        elif feature_selection == '3':
            dataset = create_melSpectogram_dataset()
            file_name = "mel_dataset.csv"
        elif feature_selection == '4':
            print("Exiting from feature extraction. Goodbye!")
            break
        else:
            print("Invalid selection. Please enter a number between 1 and 4.")
            dataset = None

        # Convert dataset to csv
        if dataset is not None:
            print("\nDataset created successfully!")
            save_option = input("\nDo you want to save this dataset to a CSV file? (yes/no): ").lower()
            if save_option == 'yes':
                try:
                    dataset.to_csv(file_name, index=False)
                    print(f"Dataset saved to {file_name}")
                except Exception as e:
                    print(f"Error saving file: {e}")