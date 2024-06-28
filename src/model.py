import pytest
import torch
from torch import nn, optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import os
import numpy as np
import random
import pandas as pd
# import fadam
import matplotlib.pyplot as plt

from sam import SAM
from tqdm import tqdm

train_loss_list=[]
train_acc_list=[]
test_loss_list=[]
test_acc_list=[]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class EFNet_L2(nn.Module):
    def __init__(self, num_classes):
        super(EFNet_L2, self).__init__()
        self.effnet = EfficientNet.from_name('efficientnet-b2')
        self.effnet._fc = nn.Linear(self.effnet._fc.in_features, num_classes)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f'Expected input to be a Tensor, but got {type(x).__name__}')
        x = self.effnet(x)
        return x

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends
set_seed(40)

@pytest.fixture
def model():
    model = EFNet_L2(num_classes=2).to(device)
    return model


def load_image_data(image_dir):
    data = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            label = 1 if 'fake' in filename else 0
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
            image = transform(image)
            data.append(image)
            labels.append(label)

    data = torch.stack(data)
    labels = torch.tensor(labels, dtype=torch.long)

    return data, labels

def data_loaders():
    train_csv = '../src/train_dataset.csv'
    test_csv = '../src/test_dataset.csv'

    # Load train data
    train_df = pd.read_csv(train_csv)
    train_labels = torch.tensor(train_df.iloc[:, 0].values, dtype=torch.long)
    train_data = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32).view(-1, 1, 28, 28)
    train_data = train_data.repeat(1, 3, 1, 1)  # Ensure 3 channels

    # Load test data
    test_df = pd.read_csv(test_csv)
    test_labels = torch.tensor(test_df.iloc[:, 0].values, dtype=torch.long)
    test_data = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32).view(-1, 1, 28, 28)
    test_data = test_data.repeat(1, 3, 1, 1)  # Ensure 3 channels

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    return train_loader, test_loader

@pytest.fixture
def optimizer_and_criterion(model):
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

#train
def train_model(train_loader, num_classes):
    model = EFNet_L2(num_classes)
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for image, labels in train_loader:
            epoch_loss = 0.0
            correct = 0
            total = 0
            images = image.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_function(output, labels)
            # First forward-backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)

            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += output.size(0)
            correct += (predicted == output).sum().item()

            # Display batch information
            print(f'Batch loss: {loss.item():.4f}, Batch accuracy: {correct / total:.4f}')
            # Calculate epoch loss and accuracy
        epoch_loss /= len(train_loader)
        accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{10}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    model_save_path = "../src/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


#test
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, labels in test_loader:
            images = image.to(device)
            labels = labels.to(device)

            output = model(images)
            predicted = torch.max(output, 1)[1]
            loss = criterion(output, labels)

            test_loss += loss.item()
            correct += (labels == predicted).sum()
            print(f"epoch {epoch + 1} - test loss: {test_loss / len(test_loader):.4f}, accuracy: {correct / len(test_loader):.4f}")
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    test_loss_list.append(int(test_loss))
    test_acc_list.append(int(100. * correct / len(test_loader.dataset)))


def load_data_from_csv(train_csv, test_csv):
    # Load train data
    train_df = pd.read_csv(train_csv)
    train_labels = torch.tensor(train_df.iloc[:, 0].values, dtype=torch.long)
    train_data = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32).view(-1, 1, 28, 28)
    train_data = train_data.repeat(1, 3, 1, 1)  # Ensure 3 channels

    # Load test data
    test_df = pd.read_csv(test_csv)
    test_labels = torch.tensor(test_df.iloc[:, 0].values, dtype=torch.long)
    test_data = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32).view(-1, 1, 28, 28)
    test_data = test_data.repeat(1, 3, 1, 1)  # Ensure 3 channels

    # Create TensorDatasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    return train_loader, test_loader

def create_model_and_optimizer():
    model = EFNet_L2(num_classes=2).to(device)
    # model.load_state_dict(torch.load('model.pth'))
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

if __name__ == "__main__":
    # Set CSV file paths
    train_csv = '../src/test_dataset.csv'
    test_csv = '../src/test_dataset.csv'

    # Load data
    train_loader, test_loader = data_loaders()
    print("non-pretrained")
    model, optimizer, criterion = create_model_and_optimizer()
    for epoch in range(1, 10):
        train_model(train_loader,2)
        evaluate_model(model,(train_loader, test_loader), (optimizer, criterion))
    plt.subplots(1, 1)
    plt.title('train_loss')
    plt.plot(np.arange(575),train_loss_list, 'k--', label='loss')
    plt.title('train_acc')
    plt.subplots(1, 2)
    plt.plot(np.arange(575),train_acc_list, 'k-',label='acc')
    plt.subplots(2, 1)
    plt.title('test_loss')
    plt.plot(np.arange(575), test_loss_list, 'k--', label='loss')
    plt.title('test_acc')
    plt.subplots(2, 2)
    plt.plot(np.arange(575), test_acc_list, 'k-', label='acc')
    plt.show()