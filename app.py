import tensorflow as tf
import streamlit as st
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os


model = load_model("model.h5")
classname = {0: "Có khối u", 1: "Không có khối u"}

def read_markdown_file(markdown_file):
    with open(markdown_file, "r", encoding='utf-8') as f:
        return f.read()

def processed_img(img_path):
    img = load_img(img_path, target_size=(96, 96, 3))
    img = img_to_array(img)
    img = img.astype('float32')
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)[0]
    y_class = output.argmax()
    result = classname[y_class]
    return result

def run():
    st.markdown("""
        <style>
            body {
                background-color: #f5f5f5;
                font-family: 'Arial', sans-serif;
            }
            .main {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                margin: auto;
            }
            .header {
                background-color: #0288d1;
                padding: 20px;
                text-align: center;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .header p {
                margin: 0;
                font-size: 1.2em;
                font-weight: 300;
            }
            h4 {
                text-align: center;
                color: #0288d1;
                font-weight: bold;
                font-size: 1.5em;
            }
            .stButton>button {
                background-color: #0288d1;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                text-align: center;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stButton>button:hover {
                background-color: #0277bd;
            }
            .footer {
                text-align: center;
                padding: 10px;
                background-color: #0288d1;
                color: white;
                border-radius: 10px;
                margin-top: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .footer p {
                margin: 5px 0;
                font-size: 1em;
            }
            .contributor {
                font-weight: bold;
                font-size: 1.1em;
            }
            .uploader {
                text-align: center;
                margin: 20px 0;
            }
            .image-container {
                text-align: center;
                margin: 20px 0;
            }
            .result-container {
                text-align: center;
                margin: 20px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header"><h1>AI NHẬN DIỆN KHỐI U Ở MÔ</h1><p>Ứng dụng AI trong y tế</p></div>', unsafe_allow_html=True)
    st.write('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h4>Phát hiện khối u qua ảnh mô học</h4>', unsafe_allow_html=True)
    st.write("**Contributors:** Nguyễn Quang Kỳ, Hoàng Trọng Sơn")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        img_file = st.file_uploader("Chọn một hình ảnh:", type=["jpg", "png"], key="fileUploader")

        if img_file is not None:
            save_image_path = './data/' + img_file.name
            if img_file.name in os.listdir('./data/'):
                os.remove(save_image_path)

            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())
            img2 = Image.open(save_image_path)
            img2 = img2.resize((200, 200))

            with col2:
                st.image(img2, use_column_width=False)
                if st.button("Nhận dạng bệnh"):
                    result = processed_img(save_image_path)
                    with col3:
                        st.success("Kết quả: " + result)
                        if result == "Có khối u":
                            khoi_u = read_markdown_file("khoi_u.md")
                            st.markdown(khoi_u, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="footer">
            <p>© 2024 AI Medical Diagnostic. All rights reserved.</p>
            <p class="contributor">Contributors:</p>
            <p>Nguyễn Quang Kỳ</p>
            <p>Hoàng Trọng Sơn</p>
        </div>
    """, unsafe_allow_html=True)

run()
