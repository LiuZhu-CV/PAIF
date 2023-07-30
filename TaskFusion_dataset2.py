# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None,label_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = './MSRS/Visible/train/MSRS/'
            data_dir_ir = './MSRS/Infrared/train/MSRS/'
            data_dir_label = './MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            data_dir_label = label_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            label = np.array(Image.open(label_path))
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )
        elif self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            # h = np.random.randint(0, 480 - 256)
            # w = np.random.randint(0, 640 - 256)
            # image_inf = image_inf[h:h + 256, w:w + 256]
            # image_vis = image_vis[h:h + 256, w:w + 256]
            image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
            )
            label = np.array(Image.open(label_path))
            ## tmp
         
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            
            # label = label[h:h + 256, w:w + 256]
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )

    def __len__(self):
        return self.length


class Fusion_dataset_Meta(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, label_path=None):
        super(Fusion_dataset_Meta, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        if split == 'train':
            data_dir_vis = './MSRS/Visible/train/MSRS/'
            data_dir_ir = './MSRS/Infrared/train/MSRS/'
            data_dir_label = './MSRS/Label/train/MSRS/'
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            data_dir_label = label_path+'Mask2/'
            data_dir_iro = label_path+'Infrared/'
            data_dir_viso = label_path+'Visible/'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
            self.filepath_viso, self.filenames_viso = prepare_data_path(data_dir_viso)
            self.filepath_iro, self.filenames_iro = prepare_data_path(data_dir_iro)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            image_inf = cv2.imread(ir_path, 0)
            label = np.array(Image.open(label_path))
            image_vis = (
                    np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.int64)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(label),
                name,
            )
        elif self.split == 'val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            vis_patho = self.filepath_viso[index]
            ir_patho = self.filepath_iro[index]
            label_path = self.filepath_label[index]
            image_vis = np.array(Image.open(vis_path))
            label = np.array(cv2.imread(label_path, 0))
            image_inf = cv2.imread(ir_path, 0)
            image_vis = np.array(Image.open(vis_path))
            image_info = cv2.imread(ir_patho, 0)
            image_viso = np.array(Image.open(vis_patho))
            h = np.random.randint(0, 480 - 256)
            w = np.random.randint(0, 640 - 256)
            image_inf = image_inf[h:h + 256, w:w + 256]
            image_vis = image_vis[h:h + 256, w:w + 256]
            image_info = image_info[h:h + 256, w:w + 256]
            image_viso = image_viso[h:h + 256, w:w + 256]
            label = label[h:h + 256, w:w + 256]
            image_vis = (
                    np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_viso = (
                    np.asarray(Image.fromarray(image_viso), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            image_info = np.asarray(Image.fromarray(image_info), dtype=np.float32) / 255.0
            image_info = np.expand_dims(image_info, axis=0)
            label = np.asarray(Image.fromarray(label), dtype=np.float32) / 255.0
            label = np.expand_dims(label, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                torch.tensor(image_viso),
                torch.tensor(image_info),
                torch.tensor(label),
                name,
            )

    def __len__(self):
        return self.length

