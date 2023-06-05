import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


"""
DocLayNet annotator:
    Input files can be save anywhere, but must be referenced accordingly.
    The usage is easy. Start the program, after loading the needed data a window will open displaying an image.
    On that image there are bounding-boxes indicating the found segmentations.
    By clicking inside the boxes this box gets marked as "read next".
    
    COCO_FOLDER: should point to the folder which holds the {train|valid|test}.json
    PNG_FOLDER: should point to the folder that holds the .png-images belonging to the entries of {train|valid|test}.json
    
    Keyboard-control:
        "q": quit the program
        "s": save the reading order ground truth which was annotated so far
        "n": going to the next image
        "i": setting an 'interest'-flag for hinting that an annotation is of interest for later purposes
"""


THIS_FILE = Path(__file__).parent

# COCO_FOLDER = (THIS_FILE / "../../DocLayNet_core/COCO").resolve()
COCO_FOLDER = (THIS_FILE / "../../Datasets/DocLayNet/DocLayNet_core/COCO").resolve()
# PNG_FOLDER = (THIS_FILE / "../../DocLayNet_core/PNG").resolve()
PNG_FOLDER = (THIS_FILE / "../../Datasets/DocLayNet/DocLayNet_core/PNG").resolve()

SAVE_FILE = (THIS_FILE / "manual_reading_order_gt.jsonl").resolve()

COLOR_ALPHA = 0.3
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3

IMAGE_FRAME_NAME = "Annotate image"
READING_ORDER = defaultdict(lambda: list())
INTEREST_FLAG = defaultdict(lambda: False)


def collision(bbox, mouse_pos):
    """
    Checks if the mouse-position is inside a given box
    :param bbox: given through the DocLayNet-dataset
    :param mouse_pos: given by the openvCV-module
    :return: True if the mouse_pos is inside a box, False otherwise
    """
    top_left, bottom_right = bbox
    mouse_x, mouse_y = mouse_pos
    return (mouse_x > top_left[0]) and (mouse_x < bottom_right[0]) and (mouse_y < top_left[1]) and (mouse_y > bottom_right[1])


def on_mouse_click(event, x, y, flags, params):
    """
    Executes on mouse click events

    :param event: Event as returned by openCV
    :param x: x-coordinate in a plane
    :param y: y-coordinate in a plane
    :param flags: additional flags for the event
    :param params: additional parameters which are optional
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{event = }, {x = }, {y = }, {flags = }, {params = }")

        for bbox_index, bbox in params.get("bboxes").items():
            if collision(bbox, (x, y)):
                READING_ORDER[params.get("file_name")].append({
                    "reading_order_position": params.get("reading_order_position"),
                    "bbox_index": bbox_index,
                    "bbox": bbox,
                })
                print(f"collision found on {bbox_index = }")
                img = params.get("image")
                shapes = np.zeros_like(img, np.uint8)
                shapes = cv2.rectangle(shapes, bbox[0], bbox[1], BOX_COLOR, cv2.FILLED)
                mask = shapes.astype(bool)
                img[mask] = cv2.addWeighted(img, COLOR_ALPHA, shapes, 1 - COLOR_ALPHA, 0)[mask]
                cv2.imshow(IMAGE_FRAME_NAME, img)

        params["reading_order_position"] += 1
        print(f"{READING_ORDER = }")


def save_reading_order():
    def repack_output_entry(key, value):
        result = dict()
        result["filename"] = key
        result["interest"] = INTEREST_FLAG[key]
        result["order"] = value
        return result

    with open(SAVE_FILE, "w", encoding="UTF-8") as save_file:
        for key, value in READING_ORDER.items():
            packed_entry = repack_output_entry(key, value)
            save_file.write(json.dumps(packed_entry))
            save_file.write("\n")
    print(f"reading order ground truth (gt) saved to: {str(SAVE_FILE)}")


def set_interest_flag(file_name):
    INTEREST_FLAG[file_name] = True


def main():
    with open(COCO_FOLDER / "train.json", "r", encoding="UTF-8") as json_fd:
        train_data = json.load(json_fd)

    annotations = train_data["annotations"]
    images = train_data["images"]

    for image in images:

        # builds the image-path
        img_path = str(PNG_FOLDER / image["file_name"])
        img = cv2.imread(img_path)
        annotation_list = list()

        bboxes = dict()
        bbox_number = 0
        reading_order_position = 0

        for annotation in annotations:
            if annotation["image_id"] != image["id"]:
                continue

            annotation_list.append(annotation["segmentation"][0])
            boxes = annotation["segmentation"][0]

            top_left = (int(round(boxes[2])), int(round(boxes[3])))
            bottom_right = (int(round(boxes[-2])), int(round(boxes[-1])))

            box = [top_left, bottom_right]
            bboxes[bbox_number] = box
            bbox_number += 1

            # builds a boxed mask for coloring on the image
            shapes = np.zeros_like(img, np.uint8)
            mask = shapes.astype(bool)
            img = cv2.rectangle(img, top_left, bottom_right, BOX_COLOR, BOX_THICKNESS)

            # adding color by the mask
            img[mask] = cv2.addWeighted(img, COLOR_ALPHA, shapes, 1 - COLOR_ALPHA, 0)[mask]

            # display the image and set a mouse-callback
            cv2.imshow(IMAGE_FRAME_NAME, img)
            on_mouse_click_params = {
                "bboxes": bboxes,
                "file_name": image["file_name"],
                "image": img,
                "reading_order_position": reading_order_position,
            }
            cv2.setMouseCallback(IMAGE_FRAME_NAME, on_mouse_click, on_mouse_click_params)

        # describes keyboard interactions
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                return
            elif key == ord('n'):
                break
            elif key == ord('s'):
                save_reading_order()
            elif key == ord('i'):
                set_interest_flag(image["file_name"])

        print(f"{image = }")
        print(f"{annotation_list = }")


if __name__ == '__main__':
    main()
