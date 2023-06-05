#- IMPORTS -#

import json
import cv2
import numpy as np
import xgboost

from collections import defaultdict
from pathlib import Path
from typing import List
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score



#- INFOS -#

COLOR_ALPHA = 0.3
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3

this_file = Path(__file__).parent.resolve()
SAVE_MODEL = this_file / "./modelle/xgboost-clf_version3.0.json"
PNG_FOLDER = this_file / "./datasets/DocLayNet"


# MAPPING
transform_class = {1: 1, 2: 10, 3: 3, 4: 9, 5: 10, 6: 10, 7: 7, 8: 11, 9: 9, 10: 10, 11: 11}
category_filter = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title"
}

category_toNum = {
    "Caption": 1,
    "Footnote": 2,
    "Formula": 3,
    "List-item": 4,
    "Page-footer": 5,
    "Page-header": 6,
    "Picture": 7,
    "Section-header": 8,
    "Table": 9,
    "Text": 10,
    "Title": 11
}

model = xgboost.XGBClassifier()
model.load_model(SAVE_MODEL)



#- FUNKTIONEN -#

def load_dataset(filename):
    """
    Loads the selfmade datasets from the /datasets folder in the same directory.
    Takes only the needed Features for the Training

    :param filename: A dataset-file from our own datasets. E.g., "train_dataset.json", "val_dataset.json"
    :return:    Returns the raw Samples from the dataset
                {"category" -> Label, "segment_box", "font_size"} -> List[Dict]
    """
    dataset = dict()

    with open(filename, "r", encoding="UTF-8") as file:
        json_file = json.load(file)
    
    for i, v in enumerate(list(json_file.values())):
        dataset[str(i)] = {
            "category": category_toNum[v["category_str"]],
            "segment_box": v["segment_box"],
            "font_size": float(v["font_informationen"]),
            "image": v["file_name"]
        }

    return dataset


def transform(train_data):
    """
    Transforms the given dataset into a for the model readable format.
    It takes a sample and creates a featurevector out of it.

    :param train_data:  given in the form of List[Dict]
                        see above in function "load_dataset"
    :return:    Returns a list with all four points of a segment_box and the font-size in it,
                together with the category index as the label.
                        
    """
    for data in train_data.values():
        data_list = data["segment_box"]
        data_list.append(data["font_size"])

        yield data_list, data["category"]-1, data["image"]


def get_class_string(indecis, label, result):
    """
    Forms a String out of the model predictions for the visualization of the output.

    :param indecis: are the indices of the prediction array for the three highest probabilities.
    :param label: is the label for this sample.
    :param result: are all the probabilities from the model to this one sample. E.g. 43.21(%)
    :returns:   a visualization of the highest three predictions of the model. 
                E.g. "Prediction: 1. 87.94 Page-footer 2. 8.63 Text 3. 1.23 List-item Label: Page-footer"
    """
    string = "Prediction: "

    if len(indecis) == 1:
        string += category_filter[indecis[0]+1]
    else:
        for i, index  in enumerate(indecis):
            string += "{i}. {perc} {category} ".format(i=i+1, perc=result[indecis[i]], category=category_filter[index+1])
    
    return string + "Label: {category}".format(category=category_filter[label+1])


def main():
    validation_data = load_dataset(this_file / "./datasets/val_dataset.json")
    transformed_dataset = transform(validation_data)

    for sample, label, image in transformed_dataset:

        # get the image
        picname = image.split(".")[0] +".png"
        img_path = str(PNG_FOLDER / picname)
        img = cv2.imread(img_path)

        top_left = (int(round(sample[0])), int(round(sample[2])))
        bottom_right = (int(round(sample[1])), int(round(sample[3])))

        # gets the models' prediction
        result = model.predict_proba([sample])[0]
        indices = np.argsort(result)[::-1]
        indices = indices[:3]
        result = [round(x*100, 2) for x in result]

        font = cv2.FONT_HERSHEY_SIMPLEX
        string = get_class_string(indices, label, result)
        print(string)

        # display the information on the document
        img = cv2.rectangle(img, top_left, bottom_right, BOX_COLOR, BOX_THICKNESS)
        img = cv2.putText(img, string, top_left, font, .5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(picname, img)

        # interactive keyboard handling
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                return
            elif key == ord('n'):
                break

            

#- PROGRAM -#


if __name__ == '__main__':
    main()
