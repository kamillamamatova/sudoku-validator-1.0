import cv2
import numpy as np
import pytesseract
import imutils

# Detects the outer Sudoku grid and returns a top down (warped) view of it
def extract_grid(image):
    # Converts image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduces noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detects edges in the image
    edged = cv2.Canny(blur, 50, 150)

    # Finds all extrernal contours
    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Compatibility fix fro OpenCV versions
    contours = imutils.grab_contours(contours)

    # Sorts contours by size, largest first
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        # Approximates polygon shape
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        # Looks for a quadrilateral, the grid
        if len(approx) == 4:
            # Reshapes to 4 corner points
            pts = approx.reshape(4, 2)
            # Warps image to top-down view
            return four_point_transform(gray, pts)
        # If no quadrilateral is found
        return None

# Warps a quadrilateral in the image to a rectangle   
def four_point_transform(image, pts):
    pts = np.array(pts, dtype = "float32")
    # Orders the points
    # Points are in clockwise order (top left, top right, bottom right, bottom left)
    rect = order_points(pts)
    # TL - top left, TR - top right, BR - bottom right, BL - bottom left
    (tl, tr, br, bl) = rect

    # Computes width and height of the new warped image
    # Difference between two 2D points is treated as a vector
    # Clockwise rectangle
    # (x, y) top left ----------- top right
    # |                                 |                         
    # |                                 |           
    # |                                 |    
    # |                                 |  
    # bottom left --------------- bottom right
    width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl))
    )
    # Calculates the vertical height of the grid by taking the longest side of the left and right vertical edges
    height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl))
    )
        
    # Matrix destination points for warp
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype = "float32") # 32-bit IEEE 754

    # Compute warp matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # Warps the image
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# Ensures the 4 grid points are consistently ordered
def order_points(pts):
    # Output array
    rect = np.zeros((4, 2), dtype = "float32")

    # Sum of (x + y)
    sum = pts.sum(axis = 1)
    # Top left, lowest sum
    rect[0] = pts[np.argmin(sum)]
    # Bottom right, highest sum
    rect[2] = pts[np.argmax(sum)]

    # Difference (x - y)
    diff = np.diff(pts, axis = 1)
    # Top right, lowest difference
    rect[1] = pts[np.argmin(diff)]
    # Bottom left, highest difference
    rect[3] = pts[np.argmax(diff)]

    return np.array(rect)

# Divides the Sudoku image into 81 cells and uses OCR to detect digits
def recognize_digits(grid_img):
    # Resizes the grid image for consistency
    grid_img = cv2.resize(grid_img, (450, 450))
    # 450 // 9 = 50
    cell_width = grid_img.shape[0] // 9
    digits = []

    for row in range(9):
        row_digits = []
        for col in range(9):
            # Calculates the coordinates of the cell
            # So if col = 1 (first loop), and each cell width is 50 pixels,
            # 0 * 50, x = 0
            # col = 1, 1 * 50, x = 50
            # so on...
            x = col * cell_width
            # y = 0, 0 * 50, y = 0
            # Waits for col loop to finish to increment row
            y = row * cell_width
            # Extracts each 50x50 cell
            # Slicing
            cell = grid_img[y: y + cell_width, x: x + cell_width]

            # Resizes up for better OCR resolution
            cell = cv2.resize(cell, (100, 10))
            # Binarizes with strong thresholding
            _, cell = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY_INV)

            # OCR w single character mode
            digit = pytesseract.image_to_string(cell, config = "--psm 8 -c tessedit_char_whitelist=123456789")
            # Removes any non digit characters
            digit = ''.join(filter(str.isdigit, digit))

            # Debugger
            print(f"Cell [{row}, {col}] OCR result: '{digit}'")

            # Converts to int or 0 if empty
            row_digits.append(int(digit) if digit else 0)
        digits.append(row_digits)
    # Returns a 9 x 9 grid of digits
    return digits

# Checks if the Sudoku grid follows the rules
def validate_sudoku(grid):
    # Stores all mistakes (row, col)
    mistakes = []
    valid = True

    def is_valid_group(group):
        # Ignores empty cells
        nums = [n for n in group if n != 0]
        # Checks for duplicates
        return len(nums) == len(set(nums))

    # Checks all rows
    for i in range(9):
        if not is_valid_group(grid[i]):
            for j in range(9):
                mistakes.append((i, j))
            valid = False
    
    # Checks all columns
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if not is_valid_group(col):
            for i in range(9):
                mistakes.append((i, j))
            valid = False

    # Checks all 3 x 3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            # y is the row index, goes down the grid
            # x is the column index, goes across the grid
            box = [grid[y][x] for y in range(i, i + 3) for x in range(j, j + 3)]
            if not is_valid_group(box):
                for y in range(i, i + 3):
                    for x in range(j, j + 3):
                        mistakes.append((y, x))
                valid = False
    
    # Overall result and specific error locations
    return valid, mistakes