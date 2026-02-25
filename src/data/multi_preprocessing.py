import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Definiamo le classi a livello globale per questo modulo
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# 1. Trasformazioni (Data Augmentation per Train, Standard per Test)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Classe PyTorch Dataset
class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}

        for cls_name in CLASSES:
            cls_path = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_path):
                for img_name in os.listdir(cls_path):
                    self.image_paths.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
