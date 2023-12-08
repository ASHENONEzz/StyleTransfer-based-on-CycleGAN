import glob
import random
from torch.utils.data import Dataset

from PIL import Image
import os

import torchvision.transforms as trForms

class ImageDataset(Dataset):
    def __init__(self, root = "",
                 transform = None,
                 model = "train"):
        self.transform = trForms.Compose(transform)

        self.pathA = os.path.join(root, model, "A/*")
        self.pathB = os.path.join(root, model, "B/*")
        self.list_A = glob.glob(self.pathA)
        self.list_B = glob.glob(self.pathB)

    def __getitem__(self, index):
        im_pathA = self.list_A[index % len(self.list_A)]
        im_pathB = random.choice(self.list_B)

        im_A = Image.open(im_pathA)
        im_B = Image.open(im_pathB)

        item_A = self.transform(im_A)
        item_B = self.transform(im_B)

        return {"A":item_A, "B":item_B}

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root = "img"

    transform_ = [trForms.Resize(256, Image.Resampling.BILINEAR), trForms.ToTensor()]
    dataloader = DataLoader(ImageDataset(root, transform_, "train"),
                            batch_size=1,
                            shuffle=True,
                            num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)
