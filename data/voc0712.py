from torch.utils.data.dataset import Dataset
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
                f'/{imgid}')[...,::-1]
        return (img,elem[1])

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
                f'/{imgid}')[...,::-1]
        return (img,elem[1])

    def __len__(self):
        return len(self.dataset)


def VOCDataset(*argv, **argc):
    return VOC2007Dataset(*argv, **argc) +\
        VOC2012Dataset(*argv, **argc)

def detect_collate(batch):
    return [i[0] for i in batch], \
            [torch.FloatTensor(i[1]) for i in batch]
