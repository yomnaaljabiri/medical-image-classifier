from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

def get_transforms(phase="train"):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

class XRayDataset(Dataset):
    CLASSES = ["NORMAL", "PNEUMONIA", "COVID19"]
    
    def __init__(self, root_dir, phase="train"):
        self.transform = get_transforms(phase)
        self.samples = []
        for label, cls in enumerate(self.CLASSES):
            folder = os.path.join(root_dir, phase, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(folder, fname), label))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label