import argparse
import logging

import cv2
import imutils
from sudoku import Sudoku
from tensorflow.python.keras.models import load_model

from model.training import train_model
from ocr.SudokuOCR import find_sudoku, extract_digit, get_board

logging.basicConfig(level=logging.INFO)

def main():
    if args.train:
        logging.info("Training Model")
        model = train_model(args.model)
    else:
        logging.info("Loading Model")
        try:
            model = load_model(args.model)
        except IOError:
            logging.error("Model not found. Train a new one with -t")
            return 0

    logging.info("Loading Image")
    img = cv2.imread(args.image)

    logging.info("Preprocessing Image")
    img = imutils.resize(img, width=600)

    logging.info("Looking for Sudoku")
    sudoku_img = find_sudoku(img, vis=args.verbose, debug=args.debug)

    logging.info("Looking for digits")
    board = get_board(sudoku_img, model)
    su = Sudoku(3, 3, board=board.tolist())
    su.show_full()

    logging.info("Solving Sudoku")
    result = su.solve()
    result.show_full()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='digit_classifier', help='name of model')
    parser.add_argument('-i', '--image', required=True, help='path to image')
    parser.add_argument('-t', '--train', action='store_true', help='train a new model')
    parser.add_argument('-v', '--verbose', action='store_true', help='visualize main steps')
    parser.add_argument('-d', '--debug', action='store_true', help='visualize every step')
    args = parser.parse_args()

    main()

