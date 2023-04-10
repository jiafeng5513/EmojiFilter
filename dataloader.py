import json
from utils import *
from torch.utils.data import dataset
from PIL import Image
import numpy as np


class dataloader(dataset.Dataset):
    def __init__(self, json_path, train=True):
        super(dataloader, self).__init__()
        self.dataset_root, self.dataset_len, self.images = self.read_dataset_json(json_path)
        self.train = train

        # train预处理
        self.train_transforms = TRAINING_TRANS

        # test预处理
        self.test_transforms = INFERENCE_TRANS
        pass

    def read_dataset_json(self, json_path):
        if not os.path.exists(json_path) or not os.path.isfile(json_path):
            raise RuntimeError("{} is not exists or it is not a regular file!".format(json_path))
        dataset_root = os.path.dirname(os.path.abspath(json_path))
        with open(json_path, "r") as json_file:
            dataset = json.load(json_file)
        dataset_root = dataset['root']
        dataset_len = dataset['len']
        images = dataset['images']
        return dataset_root, dataset_len, images

    def __getitem__(self, index):
        if index not in range(len(self.images)):
            raise RuntimeError("image index out of range!")
        image_item = self.images[index]

        item_path = os.path.join(self.dataset_root, image_item['path'].lstrip('/'))
        if not os.path.exists(item_path) or not os.path.isfile(item_path):
            raise RuntimeError("{} is not exists or it is not a regular file!".format(item_path))

        item_label = image_item['label']
        if item_label not in range(CLASSES_COUNT):
            raise RuntimeError("label = {} of {} is out of range!".format(item_label, item_path))

        img = Image.open(item_path)
        img = img.convert("RGB")

        label = int(item_label)

        feature_vec = np.array([image_item['image_size'][0], image_item['image_size'][1], image_item['file_size']],
                               dtype=np.float32)

        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        return img, label, feature_vec

    def __len__(self):
        return self.dataset_len
