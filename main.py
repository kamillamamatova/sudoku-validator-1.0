import streamlit as st
import cv2
import numpy as np
from PIL import Image
from validator import extract_grid, validate_sudoku
from recognizer import recognize_digits_from_grid

st.set_page_config(page_title = "Soduku Validator", layout = "centered")
st.title("Sudoku Validator")
st.write("Upload a photo of your solved Sudoku puzzle to check if it's valid.")

uploaded_file = st.file_uploader("Upload Sudoku Image", type = ["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # Converts the image into a NumPy array for processing
    image_np = np.array(image)
    st.image(image_np, caption = "Uploaded Sudoku")

    with st.spinner("Processing..."):
        grid_image = extract_grid(image_np)
        if grid_image is None:
            st.error("Could not detect a valid Sudoku grid.")
        else:
            digits = recognize_digits_from_grid(grid_image)

            # Confirms if it's saying "Valid Sudoku" bc OCR got it wrong
            # or the validation logic missed something
            st.write("OCR Detected Grid:")
            for row in digits:
                st.write(row)

            # Debugging
            st.write("OCR Grid:")
            for row in digits:
                st.write(row)

            valid, mistakes = validate_sudoku(digits)
            st.write("Valid Sudoku!" if valid else "Sudoku is incorrect.")
            if not valid:
                st.write("Mistakes found at:")
                for i, j in mistakes:
                    st.write(f"Row {i + 1}, Column {j + 1}")
