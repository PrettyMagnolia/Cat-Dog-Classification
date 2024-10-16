import os

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


def get_label(filename):
    # 从文件名中提取类别，根据文件名的前缀判断类别
    if 'cat' in filename:
        return 0
    elif 'dog' in filename:
        return 1
    else:
        raise ValueError(f"Unknown label in filename {filename}")


class CatDogDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data_dir = root
        self.image_files = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])

        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            # 跳过当前图像，返回下一个图像
            return self.__getitem__(idx + 1)
        label = get_label(self.image_files[idx])
        if self.transform:
            image = self.transform(image)

        return image, label
