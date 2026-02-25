from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    # Definisce le trasformazioni per il caricamento delle immagini
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    # Crea un DataLoader per il dataset di immagini
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_transforms()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader, dataset.class_to_idx
