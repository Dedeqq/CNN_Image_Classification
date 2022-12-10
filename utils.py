import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, df, images_directory):
        self.df = df
        self.directory = images_directory

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_label = self.df.iloc[idx]['name_label']
        label = self.df.iloc[idx]['label']
        number = self.df.iloc[idx]['number']
        target_iamge = torch.ones(3, 150, 150)
        loaded_image = torchvision.io.read_image(f"{self.directory}/{name_label}/{number}")
        target_iamge[:, :loaded_image.shape[1], :loaded_image.shape[2]] = loaded_image
        return target_iamge/255, label


def show(img, title=None):
    img = img.detach()
    img = torchvision.transforms.functional.to_pil_image(img)
    plt.imshow(np.asarray(img))
    if title:
        plt.title(title)
    plt.show()
