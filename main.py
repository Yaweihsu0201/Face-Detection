# main.py

import argparse

from detector import FaceDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output image")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = FaceDetector()
    detector.save_result(args.input, args.output)


if __name__ == "__main__":
    main()