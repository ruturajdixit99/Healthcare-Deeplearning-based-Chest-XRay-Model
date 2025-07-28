import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
data_dir = r"D:/Projects/Healthcare/DL/Dataset/chest_xray"
batch_size = 32
num_epochs = 10
lr = 1e-4
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Data transforms
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485],[0.229])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485],[0.229])
])

# Datasets & loaders
train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)
test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), val_tf)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# Model
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"EPOCH {epoch+1}/{num_epochs} training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "D:/Projects/Healthcare/DL/modelchest_xray_model.pth")
        print("Saved best model")

print(f"Best validation accuracy: {best_acc:.4f}")
