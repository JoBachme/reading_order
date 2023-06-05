from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import COLORS, FIGURE_SCALE, WORDS, ROWS, PARAGRAPHS, OFFSET, IMAGE_PATH


def visualize_readingbank_document(doc,
                                   paragraphs_affiliation=None,
                                   title="Clustering",
                                   show_paragraph_affiliation_in_rows=True,
                                   save_as_file=False,
                                   show_plot=True):
    """
    Displays the clustering of words into rows and rows into paragraphs

    :param doc: document which should be displayed. Has the form: {"words":[...], "rows":[...], "paragraphs":[...]}
    :param paragraphs_affiliation: classes produced by the agglomerative clustering
    :param title: title of the displayed document
    :param show_paragraph_affiliation_in_rows: if True, shows the affiliation to paragraphs in the rows sub-figure
    :param save_as_file: if True, file is written to disk
    :param show_plot: if True, shows the plot
    """
    if show_paragraph_affiliation_in_rows:
        assert paragraphs_affiliation is not None, "'paragraph_affiliation' must be given when you want to display it " \
                                                   "in the visualization"

    plots = 3
    y_scale = FIGURE_SCALE
    x_scale = plots * FIGURE_SCALE
    fig, axes = plt.subplots(1, plots, figsize=(x_scale, y_scale))

    fig.suptitle(title)
    subplot_names = [WORDS, ROWS, PARAGRAPHS]

    word_boxes = doc[WORDS]
    line_boxes = doc[ROWS]
    paragraph_boxes = doc[PARAGRAPHS]

    def draw_boxes(box_list, color, axis_no, show_paragraphs=False):
        for box_number, bbox in enumerate(box_list):
            if box_number == len(box_list):
                break
            x0, y0, x1, y1 = bbox
            w = x1 - x0
            h = y1 - y0
            try:
                facecolor = COLORS[paragraphs_affiliation[
                    box_number]] if show_paragraphs and paragraphs_affiliation is not None else "none"
            except IndexError:
                facecolor = "none"
            rectangle = plt.Rectangle((x0, y0), w, h, ec=color, fc=facecolor)
            axes[axis_no].add_patch(rectangle)

    draw_boxes(word_boxes, "cyan", 0)
    draw_boxes(line_boxes, "magenta", 1, show_paragraph_affiliation_in_rows)
    draw_boxes(paragraph_boxes, "orange", 2)

    word_boxes = np.array(word_boxes)
    doc_boxes_max_values = word_boxes.max(axis=0)
    max_x = doc_boxes_max_values[2] + OFFSET
    max_y = doc_boxes_max_values[3] + OFFSET

    doc_boxes_min_values = word_boxes.min(axis=0)
    min_x = doc_boxes_min_values[0] - OFFSET
    min_y = doc_boxes_min_values[1] - OFFSET

    for i, ax in enumerate(axes):
        # set names in sub-figures
        ax.set_title(subplot_names[i])

        ax.axis("off")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)

    plt.tight_layout()

    if save_as_file:
        save_path = Path(IMAGE_PATH)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{title}.png")

    if show_plot:
        plt.show()
