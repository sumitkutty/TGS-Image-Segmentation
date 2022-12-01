from PIL import Image
from torch.utils.data import Dataset


class segmentationdataset(Dataset):
    def __init__(self, img_paths, mask_paths, transforms= None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)