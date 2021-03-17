from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

def get_data_iter(image_path, transform, batch_size):
    data_iter = DataLoader(ImageFolder(image_path, transform=transform), batch_size, shuffle=True
        )
    return data_iter