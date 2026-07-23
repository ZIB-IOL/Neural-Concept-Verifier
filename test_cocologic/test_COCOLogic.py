import json
import torchvision.transforms as transforms
from generate_COCOLogic import LogicalCOCODataset, split_dataset

def load_category_mapping(annotation_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    return {cat['id']: cat['name'] for cat in categories}

category_map = load_category_mapping('datasets/coco/annotations/instances_train2017.json')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = LogicalCOCODataset(
    annotation_file='datasets/coco/annotations/instances_train2017.json',
    image_dir='datasets/coco/images/train2017/',
    category_id_to_name=category_map,
    transform=transform,
    filter_no_labels=True,
    exclusive_label=True,
    exclusive_match_only=True,
    log_statistics=True
)
# dataset = LogicalCOCODataset(
#     annotation_file='datasets/coco/annotations/instances_train2017.json',
#     image_dir='datasets/coco/images/train2017/',
#     category_id_to_name=category_map,
#     transform=transform,
#     assign_single_label=True, 
#     label_assignment_strategy="frequency_inverse",
#     filter_no_labels=True, 
#     log_statistics=True
# )

train_dataset, test_dataset = split_dataset(dataset)