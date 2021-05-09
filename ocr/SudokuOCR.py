from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
from skimage.segmentation import clear_border
from tensorflow.python.keras.preprocessing.image import img_to_array


def find_sudoku(img, vis=False, debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Threshold Image", thresh)
        cv2.waitKey(0)
        cv2.startWindowThread()

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzle_cnts = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        if len(approx) == 4:
            puzzle_cnts = approx
            break

    if puzzle_cnts is None:
        raise Exception("Could not find Sudoku")

    if vis or debug:
        output = img.copy()
        cv2.drawContours(output, [puzzle_cnts], -1, (0,0,255), 2)
        cv2.imshow("Outline", output)
        cv2.waitKeyEx(0)
        cv2.startWindowThread()

    puzzle = four_point_transform(img, puzzle_cnts.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_cnts.reshape(4, 2))

    if vis:
        cv2.imshow("Sudoku", warped)
        cv2.waitKeyEx(0)
        cv2.startWindowThread()

    return warped

def extract_digit(cell, debug=False):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    if debug:
        cv2.imshow('Cell', thresh)
        cv2.waitKeyEx(0)
        cv2.startWindowThread()

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    (h, w) = thresh.shape
    percent_filled = cv2.countNonZero(mask) / float(w * h)

    if percent_filled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imshow('Digit', digit)
        cv2.waitKeyEx(0)
        cv2.startWindowThread()

    return digit


def get_board(sudoku_img, model):
    board = np.zeros((9, 9), dtype='int')

    stepX = sudoku_img.shape[1] // 9
    stepY = sudoku_img.shape[0] // 9

    for y in range(9):
        row = []

        for x in range(9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            row.append((startX, startY, endX, endY))

            cell = sudoku_img[startY:endY, startX:endX]
            digit = extract_digit(cell)

            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred

    return board