import librosa
import numpy as np
import os
import pandas as pd
from feature_extraction import create_mfcc_dataset, extract_mfcc_feature_vector, extract_features_mel, audio_augmentation, preprocess_dataset

mfcc_test_features = []
mfcc_test_audio_aug_features = []
mel_test_features = []
test_ragas = []

def extract_label(file_path):
    label = os.path.splitext(os.path.basename(file_path))[0]
    return label

def modify_dataset(dataset: pd.DataFrame):
    if 'Ragas' in dataset.columns:
        dataset['Ragas'] = dataset['Ragas'].str.replace('\d+', '', regex=True)
        dataset['Ragas'] = dataset['Ragas'].str.capitalize()
        dataset['Ragas'] = dataset['Ragas'].replace('Bhoop', 'Bhoopali')
        dataset['Ragas'] = dataset['Ragas'].replace('Dkanada', 'Darbari')
    else:
        print("Warning: Required column not found in the dataset. Skipping modifications.")
    return dataset

def modify_columns(dataset: pd.DataFrame, prefix: str) -> pd.DataFrame:
    new_columns = [prefix + str(col) for col in dataset.columns[:-1]]
    dataset.columns = new_columns + [dataset.columns[-1]]
    return dataset

audio_folder = 'Test'

for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                try:
                    labels = extract_label(file_path)
                    y, sr = librosa.load(file_path, duration=30)
                    ad = audio_augmentation()
                    y_stretched = ad.time_stretch(y)
                    y_shifted = ad.pitch_shift(y_stretched, sr)
                    aug_audio = ad.noise_addition(y_shifted)
                    mfcc_feature_vector = extract_mfcc_feature_vector(y, sr)
                    feature_vector_audio_aug = extract_mfcc_feature_vector(y_shifted, sr)
                    feature_vector_mel = extract_features_mel(y, sr)
                    mfcc_test_features.append(mfcc_feature_vector)
                    mfcc_test_audio_aug_features.append(feature_vector_audio_aug)
                    test_ragas.append(labels)
                    mel_test_features.append(feature_vector_mel)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

## Convert the lists to a Pandas DataFrame
#Mel spectogram test dataset
mel_feature_df = preprocess_dataset(mel_test_features)
test_mel_ragas_df = pd.DataFrame({'Ragas': test_ragas})
test_mel_dataset = pd.concat([mel_feature_df, test_mel_ragas_df], axis=1)
test_mel_dataset = modify_columns(test_mel_dataset, "mel_")
test_mel_dataset = modify_dataset(test_mel_dataset)

# # MFCC test dataset
test_feature_df = pd.DataFrame(mfcc_test_features)
test_ragas_df = pd.DataFrame({'Ragas': test_ragas})
test_dataset = pd.concat([test_feature_df, test_ragas_df], axis=1)
test_dataset = modify_columns(test_dataset, "mfcc_")
test_dataset = modify_dataset(test_dataset)

# #MFCC audio augmented test dataset
test_audio_aug_feature_df = pd.DataFrame(mfcc_test_audio_aug_features)
test_audio_aug_feature_dataset = pd.concat([test_audio_aug_feature_df, test_ragas_df], axis=1)
test_audio_aug_feature_dataset = modify_columns(test_audio_aug_feature_dataset, "mfcc_")
test_audio_aug_feature_dataset = modify_dataset(test_audio_aug_feature_dataset)
print("\nDataset created successfully!")


# test_dataset.to_csv("mfcc_test_dataset.csv", index=False)
test_mel_dataset.to_csv("mel_test_dataset.csv", index=False)
# test_audio_aug_feature_dataset.to_csv("mfcc_test_dataset_aug.csv", index=False)

