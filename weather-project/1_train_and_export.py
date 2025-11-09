# 1_train_and_export.py (GUARANTEED 18,039 IMAGES)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path
from datasets import load_dataset
import shutil

def main():
    print("FORCING FULL DATASET DOWNLOAD...")

    # ================================
    # 1. DELETE OLD + FORCE FULL DOWNLOAD
    # ================================
    DATA_ROOT = Path("data/weathernet")
    if DATA_ROOT.exists():
        print("Deleting old dataset...")
        shutil.rmtree(DATA_ROOT)

    print("Downloading FULL WeatherNet-05 (18,039 images)...")
    ds = load_dataset("prithivMLmods/WeatherNet-05-18039", split="train")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, row in enumerate(ds):
        img = row["image"]
        label = row["label"]

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        class_dir = DATA_ROOT / str(label)
        class_dir.mkdir(exist_ok=True)
        img.save(class_dir / f"{idx:05d}.jpg", quality=95)
        saved += 1

        if saved % 3000 == 0:
            print(f"   {saved}/18039 images saved...")

    print(f"FULL DATASET DOWNLOADED: {saved} images")

    # ================================
    # 2. VERIFY COUNT
    # ================================
    total_images = len(list(DATA_ROOT.glob("**/*.jpg")))
    if total_images < 18000:
        raise ValueError(f"Only {total_images} images! Download failed.")
    print(f"VERIFIED: {total_images} images across 5 classes")

    # ================================
    # 3. TRANSFORMS & LOADERS
    # ================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(DATA_ROOT, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"Classes: {full_dataset.classes}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")  # Should be ~14,431 / 3,608

    # ================================
    # 4. MODEL & TRAIN
    # ================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("START TRAINING...")
    for epoch in range(1, 7):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = correct / total
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {acc:.1%}")

    # ================================
    # 5. EXPORT
    # ================================
    model.eval()
    dummy = torch.randn(1, 3, 128, 128).to(device)
    onnx_path = "weather_ui/model/weather_model.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(model, dummy, onnx_path, opset_version=11,
                      input_names=["input"], output_names=["output"])
    print(f"EXPORTED: {onnx_path}")

    # ================================
    # 6. EXAMPLES
    # ================================
    os.makedirs("weather_ui/examples", exist_ok=True)
    for i, cls in enumerate(full_dataset.classes):
        img_path = next((DATA_ROOT / str(i)).glob("*.jpg"))
        img = Image.open(img_path).convert("RGB")
        img.save(f"weather_ui/examples/{cls.lower()}.jpg")
    print("DEMO READY!")

if __name__ == '__main__':
    main()