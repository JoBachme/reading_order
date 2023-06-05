import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from matplotlib import pyplot as plt



# THIS_FILE = Path(__file__).parent
COCO_FOLDER = Path("./../../Datasets/DocLayNet/DocLayNet_core/COCO").resolve()
PNG_FOLDER = Path("./../../Datasets/DocLayNet/DocLayNet_core/PNG").resolve()
EXTRA_FOLDER = Path("./../../Datasets/DocLayNet/DocLayNet_extra/JSON").resolve()

SAVE_FILE = Path("./manual_reading_order_gt.jsonl").resolve()

COLOR_ALPHA = 0.3
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3

IMAGE_FRAME_NAME = "Annotated image"
READING_ORDER = defaultdict(lambda: list())

# MAPPING
transform_class = {1: 1, 2: 10, 3: 3, 4: 9, 5: 10, 6: 10, 7: 7, 8: 11, 9: 9, 10: 10, 11: 11}
WIDTH, HEIGHT = 1025, 1025



def main(n: int = 10):
    with open(COCO_FOLDER / "train.json", "r", encoding="UTF-8") as json_fd:
        train_data = json.load(json_fd)

        annotations = train_data["annotations"]
        categories = train_data["categories"]
        images = train_data["images"]

    with open(SAVE_FILE, "r", encoding="UTF-8") as json_fd:
        json_list = list(json_fd)

    for json_str in json_list:

        reading_order = json.loads(json_str)["order"]
        interested = json.loads(json_str)["interest"]

        img_path = str(PNG_FOLDER / json.loads(json_str)["filename"])
        img = cv2.imread(img_path)

        image_id = [x["id"] for x in images if x["file_name"] == json.loads(json_str)["filename"]][0]
        category_id = [x["category_id"] for x in annotations if x["image_id"] == image_id]

        for i, box in enumerate(reading_order):

            top_left = tuple(int(val) for val in box['bbox'][0])
            bottom_right = tuple(int(val) for val in box['bbox'][1])
        
            category = [x["name"] for x in categories if x["id"] == category_id[i]][0]
            text = str(box['reading_order_position']) +". - "+ category
            font = cv2.FONT_HERSHEY_SIMPLEX

            shapes = np.zeros_like(img, np.uint8)
            # shapes = cv2.rectangle(shapes, top_left, bottom_right, box_color, cv2.FILLED)
            mask = shapes.astype(bool)

            img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            img = cv2.putText(img, text, top_left, font, .5, (0, 0, 0), 1, cv2.LINE_AA)
            img[mask] = cv2.addWeighted(img, COLOR_ALPHA, shapes, 1 - COLOR_ALPHA, 0)[mask]
            cv2.imshow(IMAGE_FRAME_NAME, img)
            
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                return
            elif key == ord('n'):
                break



if __name__ == '__main__':
    main()