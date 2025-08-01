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

    # A margin to remove the grid lines from the cell image
    margin = 5

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
            cell_img = grid_img[y + margin: y + cell_size - margin, x + margin: x + cell_size - margin]

            # Preprocesses cells
            cell_processed = preprocess_cell(cell_img)

            # Marks as 0 if mostly blank
            if np.count_nonzero(cell_processed) < 100:
                row_digits.append(0)
                continue

            # Flattens and predicts
            cell_flat = cell_processed.reshape(1, -1)
            prediction = model.predict(cell_flat)[0]
            row_digits.append(int(prediction))
        digits.append(row_digits)

    return digits

def preprocess_cell(cell):
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    def center_and_resize(cell):
        # Finds bounding box of the digit
        coordinates = cv2.findNonZero(cell)
        if coordinates is None:
            return np.zeros((28, 28), dtype = np.uint8)
        x, y, w, h = cv2.boundingRect(coordinates)
        digit = cell[y: y + h, x: x + w]

        # Makes it square
        size = max(w, h)
        square = np.zeros((size, size), dtype = np.uint8)
        square[(size - h)//2: (size - h)//2 + h, (size - w)//2: (size - w)//2 + w] = digit

        # Resizes to 28 x 28
        return cv2.resize(square, (28, 28), interpolation = cv2.INTER_AREA)
    
    cell = center_and_resize(cell)

    # Binarizes with strong thresholding for robustness
    cell = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return cell.flatten() / 255.0