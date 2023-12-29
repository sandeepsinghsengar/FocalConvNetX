from pymongo import MongoClient
import torchvision.transforms as transforms
from PIL import Image
import os
import torch

def get_available_classes(root_path):
    return [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

train_classes = get_available_classes('/content/drive/MyDrive/kvasir-capsule/official_splits/1')
val_classes = get_available_classes('/content/drive/MyDrive/kvasir-capsule/official_splits/2')

class ImageFolderWithPaths(torch.utils.data.Dataset):
    def __init__(self, mongo_collection, image_root, transform=None, allowed_classes=None, class_to_idx=None):
        self.mongo_collection = mongo_collection
        self.image_root = image_root
        self.transform = transform

        if allowed_classes:
            self.image_list = list(mongo_collection.find({'label': {'$in': allowed_classes}}))
        else:
            self.image_list = list(mongo_collection.find({}))

        # Use the provided class_to_idx if available, otherwise create a new one
        self.class_to_idx = class_to_idx if class_to_idx is not None else self._create_class_to_idx_mapping()
        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        original_index = index
        while True:
            try:
                image_info = self.image_list[index]
                image_path = os.path.join(self.image_root, image_info['label'], image_info['filename'])
                image = Image.open(image_path)

                if self.transform:
                    image = self.transform(image)

                label = image_info['label']
                label_idx = self.class_to_idx.get(label, -1)

                return image, label_idx, image_path
            except FileNotFoundError:
                index = (index + 1) % len(self.image_list)
                if index == original_index:
                    raise Exception("No valid image file found in dataset.")

    def _create_class_to_idx_mapping(self):
        labels = set(item['label'] for item in self.image_list)
        class_to_idx = {label: idx for idx, label in enumerate(labels)}
        return class_to_idx