from torch.utils.data import Dataset
import torch
import cv2

class KittiDataset(Dataset):

    def __init__(self, img_path, mask_path, X, resize=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.resize = resize

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png',
                          cv2.IMREAD_GRAYSCALE)
        if self.resize:
            img = cv2.resize(img, self.resize)
            mask = cv2.resize(mask, self.resize)

        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return img, mask

    def __len__(self):
        return len(self.X)
