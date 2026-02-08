import torch
import numpy as np

class MmWhsDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, file_root, transform = None):
        self.file_list = file_list
        self.file_root = file_root
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        slice_number = self.file_list[index]

        img_slice_path = f"{self.file_root}/data/images/{slice_number}.npy"
        label_slice_path = f"{self.file_root}/data/labels/{slice_number}.npy"

        one_slice = np.load(img_slice_path)
        one_label = np.load(label_slice_path)
        
        # adding a channel so shape is (channels, width, height)
        one_slice = one_slice.reshape(1, one_slice.shape[0], one_slice.shape[1])
        one_label = one_label.reshape(1, one_label.shape[0], one_label.shape[1])

        # applying transform
        if self.transform is not None:
            one_slice = self.transform(one_slice)

        return torch.from_numpy(one_slice).float(), torch.from_numpy(one_label).float()

# transformation on images

class ImageNormalization:
    def __init__(self, level=200.0, width=600.0):
        self.level = float(level)
        self.width = float(width)
        self.a_min = level - width/2.0
        self.a_max = level + width/2.0

    def __call__(self, x):
        img = np.clip(x, self.a_min, self.a_max) # removing values above or below max and min
        img = (img - self.a_min)/(self.a_max - self.a_min)
        return img.astype(np.float32)

