import cv2
import numpy as np
import joblib

# Loads the KNN model
model = joblib.load("knn_digit_model.pkl")

def recognize_digits_from_grid(grid_img):
    # Resizes the grid image for consistency
    grid_img = cv2.resize(grid_img, (450, 450))
    # 450 // 9 = 50
    cell_size = grid_img.shape[0] // 9
    digits = []

    for row in range(9):
        row_digits = []
        for col in range(9):
            # Calculates the coordinates of the cell
            # So if col = 1 (first loop), and each cell width is 50 pixels,
            # 0 * 50, x = 0
            # col = 1, 1 * 50, x = 50
            # so on...
            x = col * cell_size
            # y = 0, 0 * 50, y = 0
            # Waits for col loop to finish to increment row
            y = row * cell_size
            # Extracts each 50x50 cell
            # Slicing
            cell = grid_img[y: y + cell_size, x: x + cell_size]

            # Preprocesses cells
            cell = preprocess_cell(cell)

            # Marks as 0 if mostly blank
            if np.count_nonzero(cell) < 100:
                row_digits.append(0)
                continue

            # Flattens and predicts
            cell_flat = cell.reshape(1, -1)
            prediction = model.predict(cell_flat)[0]
            row_digits.append(int(prediction))
        digits.append(row_digits)

    return digits

def preprocess_cell(cell):
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # Resises to 28 x 28 for KNN
    cell = cv2.resize(cell, (28, 28))
    # Binarizes with strong thresholding
    _, cell = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY_INV)
    return cell.flatten() / 255.0