import matplotlib.pyplot as plt
import os
import shutil
import librosa
import librosa.display
from dotenv import load_dotenv
from sklearn.utils import shuffle
import torchvision
import torch
from PIL import Image
import numpy as np
import random
import csv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# #train / test classification
# # 환경 변수 'fake'에서 폴더 경로
# folder_path = os.getenv('fake')
# # 폴더 내의 파일 목록
# files = os.listdir(folder_path)
# # .wav 파일이 아닌 파일만 필터링
# filtered_files = [file for file in files if not file.lower().endswith('.wav')]
# print(filtered_files)
# paths=[]
# for i in filtered_files:
#     path = os.path.join(os.getenv('fake'), i)
#     print(path)
#     paths.append(path)
#     for j in paths:
#         files = os.listdir(j)
#         for file in files:
#             print("end -------------------")
#             src = os.path.join(j, file)
#             dst = os.path.join(os.getenv('fake'), file)
#             shutil.move(src, dst)

#making_spectrogram function
def save_spectrogram(path, save):
    audio, sample_rate = librosa.load(path)
    n_fft = 2048  # 창 크기
    hop_length = n_fft // 4  # 홉 크기
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    mel_spect = librosa.power_to_db(S, ref=np.max)

    mel_spect = torch.tensor(mel_spect).to(device)

    dir_path = save
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.basename(path).replace('.wav', '.png')
    save_path = os.path.join(dir_path, file_name)

    mel_spect = mel_spect.cpu().numpy()
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(mel_spect, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    print(save_path)
    plt.savefig(save_path)
    plt.close()

def shuffle_dataset(fake_dir, real_dir,train_test):

    # Get the list of files in each directory
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, f))]
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]

    # Label the files and combine them into a single list
    labeled_data = [(f, 'fake') for f in fake_files] + [(f, 'real') for f in real_files]

    # Separate paths and labels
    paths, labels = zip(*labeled_data)

    # Shuffle the paths and labels
    shuffled_paths, shuffled_labels = shuffle(paths, labels)

    # Check the result
    for path, label in zip(shuffled_paths[:10], shuffled_labels[:10]):  # Print the first 10 elements as a sample
        print(f"Path: {path}, Label: {label}")

    # Optionally, move the files to a new directory structure if needed
    if(train_test == 0):
        output_dir = '../data/train_set'
    else:
        output_dir = '../data/test_set'
    os.makedirs(output_dir, exist_ok=True)
    for i, (path, label) in enumerate(zip(shuffled_paths, shuffled_labels)):
        # Define new path for each file in the output directory
        new_path = os.path.join(output_dir, f"{i}_{label}_{os.path.basename(path)}")
        shutil.copy2(path, new_path)

    print("Files have been copied and labeled successfully.")

#fake_spectrogram
# count = 1
# folder_path = os.getenv('fake')
# files = os.listdir('../data/Audio/Fake')
# print(files)
# for i in files:
#     print(f"{count}번째 폴더 진행중")
#     print(i)
#     save_spectrogram(os.path.join('../data/Audio/Fake',i),'../data/Audio/Fake_spectrogram')
#     count+=1

#real_spectrogram
# count = 1
# files = os.listdir('../data/Audio/Real')
# print(files)
# for i in files:
#     print(f"{count}번째 파일 진행중")
#     print(i)
#     save_spectrogram(os.path.join('../data/Audio/Real',i),'../data/Audio/Real_spectrogram')
#     count+=1

# files = os.listdir('../data/Audio/Real_spectrogram')
# print(files)
# count = 0
# for file in files:
#     print(f"{count}번째")
#     if(count%2 == 0):
#         print("짝")
#         src = os.path.join('../data/Audio/Real_spectrogram', file)
#         folder = '../data/Audio/test_real_dir'
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#         dst = os.path.join(folder, file)
#         shutil.move(src, dst)
#         count += 1
#     else:
#         print("노짝")
#         src = os.path.join('../data/Audio/Real_spectrogram', file)
#         folder = '../data/Audio/train_real_dir'
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#         dst = os.path.join(folder, file)
#         shutil.move(src, dst)
#         count += 1

def rename_files(directory):
    for count, filename in enumerate(os.listdir(directory)):
        # Get the file extension
        file_extension = filename.split('.')[-1]
        # Construct the new file name
        new_filename = f"train_real_{count + 1}.{file_extension}"
        # Form the full file path
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(src, dst)


def load_and_label_data(fake_dir, real_dir):
    data = []
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])

    # Load fake images
    print("fake start --------")
    for filename in os.listdir(fake_dir):
        try:
            filepath = os.path.join(fake_dir, filename)
            image = Image.open(filepath)
            image = transform(image)
            image = image.view(-1).numpy()  # Flatten the image
            label = 1  # Label for fake images
            data.append((image.tolist(), label))
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")
    print("real start -----------")
    # Load real images
    for filename in os.listdir(real_dir):
        try:
            filepath = os.path.join(real_dir, filename)
            image = Image.open(filepath)
            image = transform(image)
            image = image.view(-1).numpy()  # Flatten the image
            label = 0  # Label for real images
            data.append((image.tolist(), label))
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")
    return data


def save_to_csv(data, csv_filename):
    random.shuffle(data)  # Shuffle the data

    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        header = ['label'] + [f'pixel{i}' for i in range(28 * 28)]
        csvwriter.writerow(header)

        for image, label in data:
            row = [label] + image
            csvwriter.writerow(row)

def load_data(data_dir):
    data = []
    labels = []
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        image = Image.open(filepath)
        image = transform(image).to(device)
        data.append(image)
        label = 1 if 'fake' in filename else 0
        labels.append(label)
    return torch.stack(data).to(device), torch.tensor(labels, dtype=torch.long).to(device)

# print("shuffle train---------------------")
# shuffle_dataset('../data/Audio/train_fake_dir','../data/Audio/train_real_dir',0)
# print("shuffle test---------------------")
# shuffle_dataset('../data/Audio/test_fake_dir','../data/Audio/test_real_dir',1)
print("load_train ------------------")
train_data, train_labels = load_data('../data/train_set')
print("load_test ------------------")
test_data, test_labels = load_data('../data/test_set')
torch.save((train_data, train_labels),'train_data.pt')
torch.save((train_data, train_labels),'test_data.pt')

# 사용 예시
# directory = '../data/Audio/train_real_dir'
# rename_files(directory)
# shuffle_dataset('../')