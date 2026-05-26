import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import h5py
from PIL import Image 

from utils import *


def MultiTestSetDataLoader(args):
    data_list = None
    dataset_dir = args.path_for_LF_data + str(args.angRes) + 'x' + str(args.angRes) + '/'
    data_list = os.listdir(dataset_dir)

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name, Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.dataset_dir = args.path_for_LF_data + str(args.angRes) + 'x' + str(args.angRes) + '/'
        self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        try:
            LF_data_check = np.array(Image.open(file_name[0] + "/00_00.png")).astype('float32')/255.0
            H, W = LF_data_check.shape
            LF_data = np.zeros((self.angRes*self.angRes,H,W))
            for u in range(self.angRes):
                for v in range(self.angRes):
                    LF_data[u*self.angRes + v,:,:] = np.array(Image.open(file_name[0] + "/%02d_%02d.png"%(u,v))).astype('float32')/255.0
        except:
            with h5py.File(file_name[0], 'r') as hf:
                LF_data = np.array(hf.get('LF_y'))

        LF_data = torch.from_numpy(LF_data.astype(np.float32)).clone()
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return LF_data, LF_name

    def __len__(self):
        return self.item_num