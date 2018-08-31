import torch
from torch.utils.data.dataset import Dataset
import imgaug as ia
from imgaug import augmenters as iaa
import pickle
import cv2

class VOC2007Dataset(Dataset):
    def __init__(
            self, root, transform = None
            ):
        self.root = root
        with open(self.root+'anno_2007_trainval.bin','rb') as fp:
            self.dataset = pickle.load(fp)
        self.transform = transform

    def __getitem__(self, idx):
        elem = self.dataset[idx]
        imgid = elem[0]
        img = cv2.imread(f'{self.root}/VOCdevkit/VOC2007/JPEGImages'+
                f'/{imgid}')[...,::-1].copy()
        tf_det = self.transform.to_deterministic()
        return (tf_det.augment_image(img),\
                tf_det.augment_bounding_boxes(
                    [ia.BoundingBoxesOnImage([
                        ia.BoundingBox(*i[1:], label=i[0]) for i in elem[1]
                        ], shape=img.shape)
                    ])[0])

    def __len__(self):
        return len(self.dataset)


class VOC2012Dataset(Dataset):
    def __init__(
            self, root, transform = None
            ):
        self.root = root
        with open(self.root+'anno_2012_trainval.bin','rb') as fp:
            self.dataset = pickle.load(fp)
        self.transform = transform

    def __getitem__(self, idx):
        elem = self.dataset[idx]
        imgid = elem[0]
        img = cv2.imread(f'{self.root}/VOCdevkit/VOC2012/JPEGImages'+
                f'/{imgid}')[...,::-1].copy()
        tf_det = self.transform.to_deterministic()
        return (tf_det.augment_image(img),\
                tf_det.augment_bounding_boxes(
                    [ia.BoundingBoxesOnImage([
                        ia.BoundingBox(*i[1:], label=i[0]) for i in elem[1]
                        ], shape=img.shape)
                    ])[0])

    def __len__(self):
        return len(self.dataset)


def VOCDataset(*argv, **argc):
    return VOC2007Dataset(*argv, **argc) +\
        VOC2012Dataset(*argv, **argc)

def detect_collate(batch):
    return torch.stack([torch.FloatTensor(i[0]).permute(2,0,1) for i in batch]).cuda(), \
            [torch.FloatTensor([
                [j.x1/i[1].height, j.y1/i[1].width,
                j.x2/i[1].height, j.y2/i[1].width, j.label]
                for j in i[1].bounding_boxes
                ]).cuda() for i in batch]
