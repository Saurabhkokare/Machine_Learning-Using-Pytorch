import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

# -------------------------------
# Custom Dataset Class
# -------------------------------
class ImageDataset(Dataset):
    def __init__(self, image_dir, label_dict, transform=None):
        """
        image_dir : folder containing images
        label_dict : dictionary {image_name: label}
        transform : optional image preprocessing
        """
        self.image_dir = image_dir
        self.image_names = list(label_dict.keys())
        self.labels = [label_dict[name] for name in self.image_names]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        img = cv2.imread(img_path)  # BGR by default
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        img = cv2.resize(img, (32, 32))  # resize to 32x32

        if self.transform:
            img = self.transform(img)
        else:
            img = img.astype(np.float32) / 255.0  # normalize
            img = np.transpose(img, (2, 0, 1))  # HWC → CHW

        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, label_tensor

# -------------------------------
# CNN Model
# -------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 16 x 16 x 16

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # output: 32 x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

# -------------------------------
# Load your dataset
# -------------------------------

# Example: folder "images/" contains all images
# label_dict = {"img1.png":0, "img2.png":1, ...}

# For demo, let's assume a small dataset:
image_dir = "images"
label_dict = {}
for i, file in enumerate(os.listdir(image_dir)):
    if file.endswith(".png") or file.endswith(".jpg"):
        label_dict[file] = i % 10  # example labels 0-9

dataset = ImageDataset(image_dir, label_dict)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -------------------------------
# Training Setup
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 20

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)

    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# -------------------------------
# Save Model
# -------------------------------
torch.save(model.state_dict(), "cnn_model.pth")
print("✅ Model saved")
