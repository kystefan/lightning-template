import glob

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

class SHHQDataset(Dataset):
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
        
        original = transforms.ToTensor()(Image.open(image_path))
        targets['silhouette'] = transforms.ToTensor()(Image.open(image_path.replace('no_segment','segments')).convert('RGB'))

        if self.transform is not None:
            original = self.transform(original)
            targets['silhouette'] = self.transform(targets['silhouette'])
        
        return original, targets