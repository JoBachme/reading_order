#- IMPORTS -#

import itertools
import json
import numpy as np

from collections import defaultdict
from typing import List
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from matplotlib import pyplot as plt



#- INFOS -#

# PATHS
SAVE_MODEL = "./modelle/xgboost-clf_version3.0.json"
# SAVE_MODEL = "./modelle/xgboost-clf_version3.0_noFont.json"
DATASET = "./datasets/"

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
            "font_size": float(v["font_informationen"])
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

        yield data_list, data["category"]-1
        

def train_xgboost(transformed_train_set):
    classifier = XGBClassifier()
    X_train, y_train = list(zip(*transformed_train_set))
    classifier.fit(X_train, y_train)
    classifier.save_model(SAVE_MODEL)

    return classifier


def evaluate_classifier(classifier, label_names, valid_samples):
    # applies the validation samples to the classifier
    transformed_valid_set = transform(valid_samples)
    X_valid, y_valid = list(zip(*transformed_valid_set))
    y_hat = classifier.predict_proba(X_valid)
    
    # generates the confusion matrix
    cm = confusion_matrix(
        y_true=y_valid,
        y_pred=y_hat.argmax(axis=1)
    )
    plot_confusion_matrix(
        cm=cm,
        target_names=label_names,
        cmap=plt.get_cmap("viridis"),
        values_format=".2f",
        title=classifier.__module__,
        normalize=True
    )

    # generates additional metrics
    precision, recall, f1score, support = precision_recall_fscore_support(
        y_valid,
        y_hat.argmax(axis=1),  # only the label is wanted
        beta=1.0,              # f-beta-score with beta=1.0 is f1-score
        average="weighted",    # for label imbalances
    )
    hits_at_1 = hits_at_k(y_valid, y_hat, k=1, labels=label_names)
    hits_at_3 = hits_at_k(y_valid, y_hat, k=3, labels=label_names)
    auroc_score = roc_auc_score(y_valid, y_hat, multi_class="ovr")
    print(f"{classifier.__module__=}\n{precision=}\n{recall=}\n{f1score=}\n{support=}\n{auroc_score=}\n{hits_at_1=}\n{hits_at_3=}")


def hits_at_k(y_true, y_score, labels=None, k=1, normalized=True):
    """
    Calculates the hits@k manually.

    :param y_true: ground truth labels
    :param y_score: outputted labels by a classifier
    :param labels: all possible classes/labels
    :param k: if the correct label is in the first 'k' entries, this counts as hit
    :return: hits@k per class
    """
    classes = defaultdict(lambda: 0)
    hit_at_k = defaultdict(lambda: 0)
    for true_value, predicted_value in zip(y_true, y_score):
        sorted_predicted_probas = np.argsort(predicted_value)[::-1]  # sorting probas in descending order
        top_k_predicted_probas = sorted_predicted_probas[:k]
        classes[true_value] += 1
        hit_at_k[true_value] += np.sum(top_k_predicted_probas == true_value)  # checks if the expected class is in top-k predictions

    result_top_k_hits = dict()

    for _class in classes:
        label = labels[_class] if labels is not None else _class
        result_top_k_hits[label] = hit_at_k[_class] / classes[_class]

    return result_top_k_hits


def plot_confusion_matrix(
        cm,
        target_names,
        title='Confusion matrix',
        cmap=None,
        normalize=True,
        values_format=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Citiation:
    - http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    - https://stackoverflow.com/questions/48855290/plotted-confusion-matrix-values-overlapping-each-other-total-classes-90

    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: given classification classes such as [0, 1, 2]. The class names, for example: ['high', 'medium', 'low']
    :param title: the text to display at the top of the matrix
    :param cmap: the gradient of the values displayed from matplotlib.pyplot.com. See http://matplotlib.org/examples/color/colormaps_reference.html or plt.get_cmap('jet') or plt.cm.Blues.
    :param normalize: If False, plot the raw numbers. If True, plot the proportions
    :param values_format: Can shape the format for the numbers shown in the matrix, e.g.: '.4f'. Default: ".4f"
    """

    FONT_SIZE = 16
    FIGURE_SCALE = 2
    PRIMARY_FONT_COLOR = "black"
    SECONDARY_FONT_COLOR = "white"

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if values_format is None:
        values_format = ".4f"

    plt.figure(figsize=(8*FIGURE_SCALE, 6*FIGURE_SCALE))
    plt.title(title, fontdict={"fontsize": FONT_SIZE + 10})

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONT_SIZE)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    font_color_threshold = cm.max() / 3
    formatting_string = str("{:"+values_format+"}") if normalize else "{:,}"

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, formatting_string.format(cm[i, j]),
                 horizontalalignment="center",
                 fontsize=FONT_SIZE,
                 color=PRIMARY_FONT_COLOR if cm[i, j] > font_color_threshold else SECONDARY_FONT_COLOR)

    plt.colorbar()
    plt.ylabel("True label", fontsize=FONT_SIZE)
    plt.xlabel("Predicted label", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig("confusion-matrix-version2.png")
    plt.show()


def main():
    train_data = load_dataset(DATASET+"train_dataset.json")
    validation_data = load_dataset(DATASET+"val_dataset.json")

    transformed_train_set = transform(train_data)
    classifier = train_xgboost(transformed_train_set)
    print(classifier)

    evaluate_classifier(classifier, list(category_filter.values()), validation_data)

#- PROGRAM -#

if __name__ == '__main__':
    main()
    # Train with better parameters?
    # https://github.com/dmlc/xgboost/issues/6353