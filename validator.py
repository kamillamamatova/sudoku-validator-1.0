import cv2
import numpy as np
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

    return rect

# Checks if the Sudoku grid follows the rules
def validate_sudoku(grid):
    for i in range(9):
        row = [n for n in grid[i] if n != 0]
        col = [grid[j][i] for j in range(9) if grid [j][i] != 0]
        if len(row) != len(set(row)) or len(col) != len(set(col)):
            return False, []
        
    # Checks all 3 x 3 boxes
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = []
            for i in range(3):
                for j in range(3):
                    val = grid[box_row + i][box_col + j]
                    if val != 0:
                        box.append(val)
            if len(box) != len(set(box)):
                return False, []
                
    return True, []

    # Stores all mistakes (row, col)
    #mistakes = []
    #valid = True

    #def is_valid_group(group):
        # Ignores empty cells
        #nums = [n for n in group if n != 0]
        # Checks for duplicates
        #return len(nums) == len(set(nums))

    # Checks all rows
    #for i in range(9):
        #if not is_valid_group(grid[i]):
            #for j in range(9):
                #mistakes.append((i, j))
            #valid = False
    
    # Checks all columns
    #for j in range(9):
        #col = [grid[i][j] for i in range(9)]
        #if not is_valid_group(col):
            #for i in range(9):
                #mistakes.append((i, j))
            #valid = False

    # Checks all 3 x 3 boxes
    #for i in range(0, 9, 3):
        #for j in range(0, 9, 3):
            # y is the row index, goes down the grid
            # x is the column index, goes across the grid
            #box = [grid[y][x] for y in range(i, i + 3) for x in range(j, j + 3)]
            #if not is_valid_group(box):
                #for y in range(i, i + 3):
                    #for x in range(j, j + 3):
                        #istakes.append((y, x))
                #valid = False
    
    # Overall result and specific error locations
    # return valid, mistakes