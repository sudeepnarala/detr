import torch
import os
import pickle

class PartialDataSet(torch.utils.data.Dataset):
    def __init__(self, backbone_features_dir):
        self.backbone_features_dir = backbone_features_dir
    def __len__(self):
        return len(os.listdir(self.backbone_features_dir)) // 2
    def __getitem__(self, idx):
        samples = pickle.load(open("{}/batch{}_samples".format(self.backbone_features_dir, idx), "rb"))
        targets = pickle.load(open("{}/batch{}_targets".format(self.backbone_features_dir, idx), "rb"))
        # Tensor, Mask, Targets
        tensor = samples[0][0].tensors
        mask = samples[0][0].mask
        pos_enc = samples[1][0]
        targets = targets
        return tensor, mask, pos_enc, targets