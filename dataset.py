import glob

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class DummyDataset(Dataset):
    def __init__(self, data_path,
                 img_height, img_width):
        
        self.image_paths = [image_path for image_path in glob.glob(data_path + '/*')]
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        targets = {}
        
        original = transforms.ToTensor()(Image.open(image_path).convert('RGB'))

        if self.transform is not None:
            original = self.transform(original)

        targets['original'] = original 
        
        return original, targets