import streamlit as st
import ultralytics
import numpy as np
import pandas as pd
import pytesseract
import cv2
import re
from PIL import Image

model = ultralytics.YOLO('best.pt')

st.title("Number Plate Detection System")

uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg"])

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#dummy data
df = pd.DataFrame({
    "Plate_number": ["DL7CN5617", "MH20BQ20"],
    "Owner_Name": ["Amal", "Jose"],
    "Department": ["CSE", "EEE"],
    "Employee_id": [101, 123]
})

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(img)

    for r in results:
        boxes = r.boxes.xyxy

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            plate = img[y1:y2, x1:x2]

            st.image(plate, caption="Detected Plate")

            text = pytesseract.image_to_string(plate)
            plate_text = re.sub(r'[^A-Z0-9]', '', text)

            st.write("🔍 Extracted Text:", plate_text)

            if plate_text.upper() in df["Plate_number"].values:
                st.success("Authorized Vehicle")

                result = df[df["Plate_number"] == plate_text.upper()].iloc[0]

                st.write("Vehicle Details")
                st.write("Plate Number:", result["Plate_number"])
                st.write("Owner Name:", result["Owner_Name"])
                st.write("Department:", result["Department"])
                st.write("Employee ID:", result["Employee_id"])

            else:
                st.error("Unauthorized Vehicle")