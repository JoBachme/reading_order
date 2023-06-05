import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

from config import DEBUG, DISTANCE_THRESHOLD, CLUSTER_THRESHOLD


def build_rows(readingbank_document):
    """
    Cluster the word-boxes to line-boxes.

    :param readingbank_document: readingBank-document
    :return: dictionary of {lineNo.: linebox}
    """
    row_dict = {}
    bbox_nr = 1
    first = True
    src_list = readingbank_document["tgt"]
    max_length = len(src_list)
    curr_bottom_right, curr_top_left = [0, 0], [0, 0]

    for i, bbox in enumerate(src_list):
        top_x, top_y = int(bbox[0]), int(bbox[1])
        bottom_x, bottom_y = int(bbox[2]), int(bbox[3])

        # save the first point of a new row
        if first:
            first = False
            curr_top_left = [top_x, top_y]
            curr_bottom_right = [bottom_x, bottom_y]

        # expand the box to the right when the direct successor is further to the right side
        if curr_bottom_right[0] < bottom_x:
            curr_bottom_right = [bottom_x, bottom_y]

        if i < max_length - 1:
            thresh = int(src_list[i+1][3])

            # compare the height of the next point with the current (range from +/-3)
            # compare the width of the next point with the current (range smaller than 10 pixel)
            if curr_bottom_right[1] not in range(thresh-6, thresh+6) or int(src_list[i+1][0]) - curr_bottom_right[0] > 10:
                row_dict[bbox_nr] = [*curr_top_left, *curr_bottom_right]
                bbox_nr += 1
                first = True

    # add the final box to the dict
    row_dict[bbox_nr] = [*curr_top_left, *curr_bottom_right]
    return row_dict


def manhattan_distance(bbox_a, bbox_b, threshold, x_only=False, y_only=False, check=False):
    # indices of a bounding box (bbox):  0 |  1 |  2 |  3
    # entries of a bounding box (bbox): x0 | y0 | x1 | y1
    #
    # o--> x
    # |
    # v
    # y
    #
    # (x0,y0)
    # -------------
    # |           |
    # |     A     |     (x0,y0)
    # |           |       -------------
    # -------------       |           |
    #        (x1, y1)     |     B     |
    #                     |           |
    #                     -------------
    #                               (x1,y1)
    # manhattan distance in x-axis: x0_B - x1_A
    # manhattan distance in y-axis: y0_B - y1_A

    if check:
        if type(bbox_a) is list and type(bbox_b) is list and bbox_a == bbox_b:
            return 0

        if type(bbox_a) is np.ndarray and type(bbox_b) is np.ndarray and (bbox_a == bbox_b).all():
            return 0

    max_x_a = max(bbox_a[0], bbox_a[2])
    min_x_a = min(bbox_a[0], bbox_a[2])
    max_y_a = max(bbox_a[1], bbox_a[3])
    min_y_a = min(bbox_a[1], bbox_a[3])

    max_x_b = max(bbox_b[0], bbox_b[2])
    min_x_b = min(bbox_b[0], bbox_b[2])
    max_y_b = max(bbox_b[1], bbox_b[3])
    min_y_b = min(bbox_b[1], bbox_b[3])

    diff_x_1 = max_x_a - min_x_b
    diff_x_2 = max_x_b - min_x_a

    diff_y_1 = max_y_a - min_y_b
    diff_y_2 = max_y_b - min_y_a

    diff_x = abs(min(diff_x_1, diff_x_2))
    diff_y = abs(min(diff_y_1, diff_y_2))

    if bbox_a[0] <= bbox_b[0] <= bbox_a[2] and bbox_a[1] <= bbox_b[1] <= bbox_a[3] \
            or bbox_b[0] <= bbox_a[0] <= bbox_b[2] and bbox_b[1] <= bbox_a[1] <= bbox_b[3]:
        return 0.

    if x_only:
        return diff_x

    if y_only:
        return diff_y

    resulting_distance = diff_x if diff_y < threshold else diff_y

    if DEBUG:
        print(f"{diff_x=}")
        print(f"{diff_y=}")
        print(f"{threshold=}")
        print(f"{resulting_distance=}")
        print()

    return resulting_distance


def calculate_distance_matrix(doc, distance_threshold):
    pair_distances = pairwise_distances(doc, doc, metric=manhattan_distance, threshold=distance_threshold)
    return pair_distances


def get_paragraphs(doc, cluster_threshold, distance_threshold):
    """
    Finds the cluster the bounding boxes of word-lines correspond to.

    :param doc: a document consisting of bounding boxes of word-lines or words.
    :param cluster_threshold: threshold on which it is decided if word-line correspond to a cluster or not.
    :param distance_threshold: threshold for the manhattan-distance which decides on how close boxes can be on the y-axis until the x-axis is to be considered.
    :return: list of cluster-classes assigning all input-bounding boxes to a cluster
    """
    distance_matrix_x = calculate_distance_matrix(doc, distance_threshold)
    agg_cluster = AgglomerativeClustering(
        n_clusters=None,
        compute_full_tree=True,
        distance_threshold=cluster_threshold,
        compute_distances=True,
        linkage="single",
        metric="precomputed"
    )
    preds = agg_cluster.fit_predict(distance_matrix_x)
    return preds


def make_paragraph_from_prediction(lines, predicted_classes):
    """
    Clusters the paragraphs from the line-boxes

    :param lines: list of line-boxes
    :param predicted_classes: classes obtained from the agglomerative clustering
    :return: list of paragraph-boxes
    """
    classes = np.unique(predicted_classes)
    src_doc = np.array(lines)

    paragraph_boxes = list()

    for c in classes:

        # find for every class the lines which belong to the same paragraph
        class_mask = predicted_classes == c
        class_boxes = src_doc[class_mask]

        # find the paragraph-box
        mins = class_boxes.min(axis=0)
        maxs = class_boxes.max(axis=0)
        x0, y0 = mins[0], mins[1]
        x1, y1 = maxs[2], maxs[3]
        paragraph_boxes.append([x0, y0, x1, y1])

    return np.array(paragraph_boxes)


def process(readingbank_document):
    # words -> lines/rows
    rows_dict = build_rows(readingbank_document)
    rows = list(rows_dict.values())

    # lines/rows -> paragraphs
    paragraph_classes = get_paragraphs(rows,
                                       cluster_threshold=CLUSTER_THRESHOLD,
                                       distance_threshold=DISTANCE_THRESHOLD)
    paragraphs = make_paragraph_from_prediction(rows, paragraph_classes)

    struct = {
        "words": readingbank_document["src"],
        "rows": rows,
        "paragraphs": paragraphs,
        "paragraph_class_per_line": paragraph_classes
    }

    from visualize import visualize_readingbank_document
    visualize_readingbank_document(struct, paragraph_classes)

    return struct


