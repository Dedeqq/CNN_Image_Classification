from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, df, images_directory, transforms=None):
        self.df = df
        self.directory = images_directory
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_label = self.df.iloc[idx]['name_label']
        label = self.df.iloc[idx]['label']
        number = self.df.iloc[idx]['number']
        image = Image.open(f"{self.directory}/{name_label}/{number}")
        if self.transforms:
            image = self.transforms(image)
        return image, label
