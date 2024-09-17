import logging
import argparse
from utils import download_data

logging.basicConfig(level=logging.INFO)


def main():
    arg_parser = argparse.ArgumentParser(description="download_data.py")

    arg_parser.add_argument(
        "data_url",
        nargs="?",
        action="store",
        default="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz",
        type=str,
    )

    params = arg_parser.parse_args()

    download_data(params.data_url)


if __name__ == "__main__":
    main()
