import json
from pathlib import Path

from clustering import process
from visualize import visualize_readingbank_document


def load_documents():
    with open(Path(__file__).parent / "../data/example.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line.strip())


def example_usage():

    for readingbank_document in load_documents():

        # example usage
        clustered_document = process(readingbank_document)
        visualize_readingbank_document(
            doc=clustered_document,
            paragraphs_affiliation=clustered_document["paragraph_class_per_line"],
        )

        # stopping the process
        prompt = input("Press ENTER to continue, press 'q' and ENTER to quit:")
        if prompt == "q" or prompt == "quit":
            break


if __name__ == '__main__':
    example_usage()
