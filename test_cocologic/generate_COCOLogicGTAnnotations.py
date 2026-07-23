import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from collections import defaultdict, Counter
import random


def load_category_mapping(annotation_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    return {cat['id']: cat['name'] for cat in categories}


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


class LogicalConceptAnnotationDataset(Dataset):
    def __init__(self, annotation_file, category_id_to_name,
                 filter_no_labels=True, exclusive_label=True, exclusive_match_only=True,
                 log_statistics=False):
        """
        Returns:
            - binary category presence vector (length = num_categories)
            - logical class label (int index if exclusive_label=True, else multi-hot)
        """
        self.exclusive_label = exclusive_label
        self.exclusive_match_only = exclusive_match_only
        self.filter_no_labels = filter_no_labels

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.categories = list(sorted(set(category_id_to_name.values())))
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
        self.num_categories = len(self.categories)

        self.imgs = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']

        self.image_to_categories = defaultdict(set)
        category_frequency = Counter()
        for ann in self.annotations:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            cat_name = category_id_to_name[cat_id]
            self.image_to_categories[img_id].add(cat_name)
            category_frequency[cat_name] += 1

        # # logical class definitions
        # self.logical_classes = [
        #     ("Domestic Animal Scene", lambda cats: ('cat' in cats or 'dog' in cats) and 'person' in cats),
        #     ("Personal Transport XOR Car", lambda cats: 'person' in cats and (('bicycle' in cats) ^ ('car' in cats))),
        #     ("Urban Traffic Scene", lambda cats: any(c in cats for c in ['car', 'bus', 'truck']) and 'traffic light' in cats and 'person' in cats),
        #     ("Eating Situation", lambda cats: 'bowl' in cats and any(c in cats for c in ['spoon', 'fork', 'cup'])),
        #     ("Empty Sitting Area", lambda cats: any(c in cats for c in ['couch', 'chair']) and 'person' not in cats),
        #     ("Rural Animal Scene", lambda cats: any(c in cats for c in ['cow', 'horse', 'sheep']) and 'person' not in cats),
        #     ("Outdoor Activity XOR Vehicle", lambda cats: 'person' in cats and any(c in cats for c in ['skateboard', 'kite', 'frisbee']) and not any(c in cats for c in ['car', 'bus', 'bicycle'])),
        #     ("Child Play Area", lambda cats: 'person' in cats and any(c in cats for c in ['swing', 'slide', 'bench'])),
        #     ("Fast Food Moment", lambda cats: 'pizza' in cats or 'hot dog' in cats),
        #     ("Nature Alone", lambda cats: any(c in cats for c in ['tree', 'bird']) and 'person' not in cats),
        # ]
        self.logical_classes = [
            # 1. Conflicted Companions (Leash vs Licence). An image features either a dog or a car, but not both.
            ("Conflicted Companions (Leash vs Licence)", lambda cats: ('dog' in cats) ^ ('car' in cats)),
            # 2. Ambiguous Pairs (Pet vs Ride Paradox). The image includes either a cat or a dog (but not both), 
            # and either a bicycle or a motorcycle (but not both).
            ("Ambiguous Pairs (Pet vs Ride Paradox)", lambda cats: 
                (('cat' in cats) ^ ('dog' in cats)) and 
                (('bicycle' in cats) ^ ('motorcycle' in cats))
            ),
            # 3. Pair of Pets. Exactly two of the following animals are present: a cat, a dog, or a bird.
            ("Pair of Pets", lambda cats: 
                sum(c in cats for c in ['cat', 'dog', 'bird']) == 2
            ),
            # 4. Rural Animal Scene. The image includes one or more rural animals (cow, horse, or sheep) and no people.
            ("Rural Animal Scene", lambda cats: any(c in cats for c in ['cow', 'horse', 'sheep']) 
             and 'person' not in cats
             ),
            # 5. Animal Meet Traffic. The image contains a rural animal (horse, cow, or sheep) and a 
            # traffic-related object (car, bus, or traffic light).
            ("Animal Meet Traffic", lambda cats:
                any(c in cats for c in ['horse', 'cow', 'sheep']) and
                any(c in cats for c in ['car', 'bus', 'traffic light'])
            ),
            # 6. Home Alone. The image includes furniture (a couch or chair) and exactly one person.
            ("Home Alone", lambda cats: any(c in cats for c in ['couch', 'chair']) and 'person' in 
             cats and sum(c == 'person' for c in cats) == 1
             ),
            # 7. Empty House. The image includes indoor furniture (a couch or chair) but no person is present.
            ("Empty House", lambda cats: any(c in cats for c in ['couch', 'chair']) and 'person' not in cats),
            # 8. Odd Ride Out. Exactly one of the following is present: a bicycle, motorcycle, car, or bus.
            ("Odd Ride Out", lambda cats:
                sum(c in cats for c in ['bicycle', 'motorcycle', 'bus', 'car']) == 1
            ),
            # 9. Personal Transport XOR Car. A person is present alongside either a bicycle or a car — but not both.
            ("Personal Transport XOR Car", lambda cats: 'person' in cats and (('bicycle' in cats) ^ ('car' in cats))),
            # 10. Unlikely Breakfast Guests. The image shows a bowl (suggesting food) and at least one animal (dog, cat, horse, cow, or sheep).
            ("Unlikely Breakfast Guests", lambda cats:
                'bowl' in cats and
                any(c in cats for c in ['dog', 'cat', 'horse', 'cow', 'sheep'])
            ),
        ]

        self.class_names = [name for name, _ in self.logical_classes]
        self.image_ids = []
        self.labels = []
        self.concept_vectors = []

        for img_id in self.imgs:
            cats = self.image_to_categories.get(img_id, set())
            label_vec = [int(fn(cats)) for _, fn in self.logical_classes]

            if self.filter_no_labels and not any(label_vec):
                continue
            if self.exclusive_match_only and sum(label_vec) != 1:
                continue

            # Convert to binary concept vector
            concept_vec = [0] * self.num_categories
            for cat in cats:
                if cat in self.cat2idx:
                    concept_vec[self.cat2idx[cat]] = 1

            if self.exclusive_label:
                label = label_vec.index(1)
            else:
                label = label_vec

            self.image_ids.append(img_id)
            self.concept_vectors.append(torch.tensor(concept_vec, dtype=torch.float))
            self.labels.append(torch.tensor(label, dtype=torch.long if self.exclusive_label else torch.float))

        if log_statistics:
            print(f"LogicalConceptAnnotationDataset: {len(self.image_ids)} examples loaded.")
            for i, name in enumerate(self.class_names):
                count = sum(1 for label in self.labels if (label == i if self.exclusive_label else label[i]))
                print(f" - {name:<30}: {count}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return self.concept_vectors[idx], self.labels[idx]