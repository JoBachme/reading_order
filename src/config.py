from pathlib import Path
from functools import cache


@cache
def load_colors():
    local_color_list = list()
    with open(Path(__file__).parent.resolve() / "../data/colors.txt", "r", encoding="UTF-8") as colors_file:
        for line in colors_file:
            local_color_list.append(line.strip())
    return local_color_list


DEBUG = False

DISTANCE_THRESHOLD = 5
CLUSTER_THRESHOLD = 5

FIGURE_SCALE = 8
COLORS = load_colors()
IMAGE_PATH = "./img/"
PARAGRAPHS = "paragraphs"
ROWS = "rows"
WORDS = "words"
OFFSET = 50  # in px
