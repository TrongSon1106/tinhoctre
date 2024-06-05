import tensorflow as tf
import streamlit as st
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os

model=load_model("model.h5")
classname={0:"Có khối u", 1:"Không có khối u"}


def read_markdown_file(markdown_file):
    text=""
    with open(markdown_file, "r", encoding='utf-8') as f:
        text= f.read()
    return text
    #return Path(markdown_file,encoding='utf-8').read_text()

def processed_img(img_path):
    img=load_img(img_path,target_size=(96,96,3))
    img=img_to_array(img)
    img = img.astype('float32')
    img /= 255.0
    img=np.expand_dims(img,axis=0)
    output=model.predict(img)[0]
    print(output)
    y_class = output.argmax()
    print(y_class)
    result = classname[y_class]
    print(result)
    return result

def run():
    # logo = Image.open('medical.jpg')
    # medical_image = Image.open('logo.png')
    # st.markdown("""
    #     <div style="display: flex; justify-content: space-between;">
    #         <div>
    #             <img src="data:image/png;base64,{logo}" alt="logo" style="height: 60px;">
    #          </div>
    #     </div>
    #     """.format(logo=logo), unsafe_allow_html=True)
    st.markdown('''<h4 style='text-align: center; color: green; font-weight: bold;'>AI NHẬN DIỆN KHỐI U Ở MÔ</h4>''',
                unsafe_allow_html=True)
    #st.title("Phần mềm ứng dụng trí tuệ nhân tạo hỗ trợ nhận dạng một số bệnh về da tại nhà")
    st.write("Tác giả: Nguyễn Quang Kỳ, Hoàng Trọng Sơn")
    col1, col2, col3 = st.columns(3)
    benh=0
    with col1:
        img_file = st.file_uploader("Chọn một hình ảnh:", type=["jpg", "png"])
        
        if img_file is not None:
            save_image_path = './data/'+img_file.name
            print(os.listdir('./data/'))
            if img_file.name in os.listdir('./data/'):
                os.remove(save_image_path)
            
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())
            img2 = Image.open(save_image_path)
            img2 = img2.resize((200,200))

            with col2:
                st.image(img2,use_column_width=False)            
                if st.button("Nhận dạng bệnh"):
                    result = processed_img(save_image_path)

                    with col3:
                        st.success("Kết quả: " + result)
                        if result=="Có khối u":
                            khoi_u = read_markdown_file("khoi_u.md")
                            st.markdown(khoi_u, unsafe_allow_html=True)
                            benh=1
                            

    # if benh==1:
    #     khoi_u_vd = read_markdown_file("khoi_u_youtube.md")
    #     st.markdown(khoi_u_vd, unsafe_allow_html=True)

        
run()
