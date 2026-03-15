# convert_coco_to_yolo.py
from ultralytics.data.converter import convert_coco

# Convert COCO annotations to YOLO format
convert_coco(
    labels_dir='data/coco/annotations/',  # Where your JSON files are
    use_segments=False,  # Set to True if you need segmentation
    use_keypoints=False  # Set to True if you need keypoints
)

print("Conversion complete! Labels saved in data/coco/labels/")