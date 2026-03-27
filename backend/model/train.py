import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import MedicalClassifier
from dataset import XRayDataset
from sklearn.metrics import classification_report


def train(data_dir="../data/", epochs=20, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = XRayDataset(data_dir, "train")
    val_ds   = XRayDataset(data_dir, "val")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = MedicalClassifier(num_classes=3).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        # ── Training loop
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        
        # ── Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        
        acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} — Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved best model (acc={acc:.4f})")

if __name__ == "__main__":
    train()