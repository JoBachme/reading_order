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



def main(n: int = 10):
    with open(COCO_FOLDER / "train.json", "r", encoding="UTF-8") as json_fd:
        train_data = json.load(json_fd)

        annotations = train_data["annotations"]
        categories = train_data["categories"]
        images = train_data["images"]

        for image in images[0:]:
            img_path = str(PNG_FOLDER / image["file_name"])
            img = cv2.imread(img_path)
            annotation_list = list()

            bboxes = dict()
            bbox_number = 0

            for annotation in annotations:
                if annotation["image_id"] == image["id"]:
                    annotation_list.append(annotation["segmentation"][0])
                    boxes = annotation["segmentation"][0]

                    top_left = (int(round(boxes[2])), int(round(boxes[3])))
                    bottom_right = (int(round(boxes[-2])), int(round(boxes[-1])))

                    box = [top_left, bottom_right]
                    bboxes[bbox_number] = box
                    bbox_number += 1
                    
                    category = [x["name"] for x in categories if x["id"] == annotation["category_id"]][0]
                    
                    # Mapped Classes
                    # category = [x["name"] for x in categories if x["id"] == transform_class[annotation["category_id"]]][0]

                    text = str(bbox_number) +". - "+ category
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    shapes = np.zeros_like(img, np.uint8)
                    # shapes = cv2.rectangle(shapes, top_left, bottom_right, box_color, cv2.FILLED)
                    mask = shapes.astype(bool)
                    img = cv2.rectangle(img, top_left, bottom_right, BOX_COLOR, BOX_THICKNESS)
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