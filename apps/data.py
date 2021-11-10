import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def app():
    st.title('CVD Data')

    st.write("""
    โรคหัวใจและหลอดเลือด (cardiovascular diseases) เป็นกลุ่มโรคที่เกิดกับระบบหัวใจและหลอดเลือดซึ่งเป็นสาเหตุการเสียชีวิตลำดับต้นๆของคนไทย หากเราทำนายได้ว่าบุคคลใดมีความเสี่ยงที่จะเป็นโรคหัวใจและหลอดเลือดได้ 
    จะทำให้เกิดประโยชน์ค่อนข้างมากไม่ว่าจะเป็นการทำการรักษาที่ทันเวลา หรือการรักษาก่อนเกิดสภาวะที่ร้ายแรง โดยชุดข้อมูลที่นำมาวิเคราะห์และนำมาใช้นี้ มีข้อมูลพื้นฐานของผู้ป่วยต่างๆ เช่น เพศ อายุ การสูบบุหรี่ พฤติกรรมของผู้ป่วย โดยชุดข้อมูลที่เลือกนำมาวิเคราะห์ คือ `'cvd_dataset.csv'`
    ### แหล่งข้อมูล [Cardiovascular Disease dataset](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
    """)
    
    cvd = pd.read_csv('./data/cvd_dataset.csv', sep=';', index_col=["id"])
    st.write("ตัวอย่างข้อมูลที่นำมาวิเคราะห์")
    st.dataframe(cvd.sample(10))  

    st.write("""
    ### cvd_dataset.csv
    ข้อมูลชุดนี้มีข้อมูลทั้งหมด 70000 แถว และ 12 คอลัมน์ ซึ่งประกอบไปด้วยข้อมูลต่างๆที่เกี่ยวข้องกับผู้ป่วย โดยข้อมูลสามารถแบ่งได้เป็น 3 ประเภท ได้แก่ 

    - Objective: factual information
    - Examination: results of medical examination
    - Subjective: information given by the patient

    **คำอธิบายแต่ละคอลัมน์**

    | Column | Description | Data Type |
    | :--------- | :---------- | :----|
    | age | Age | int (days)
    | height | Height | int (cm) |
    | weight | Weight | float (kg) |
    | gender | Gender | categorical code |
    | ap_hi | Systolic blood pressure | int |
    | ap_lo | Diastolic blood pressure | int |
    | cholesterol | Cholesterol | 1: normal, 2: above normal, 3: well above normal |
    | gluc | Glucose | 1: normal, 2: above normal, 3: well above normal |
    | smoke | Smoking | binary |
    | alco | Alcohol intake | binary |
    | active | Physical activity | binary |
    | cardio | Presence or absence of cardiovascular disease | binary |
    ---------
    
    """)

    st.write("""
    **วิเคราะห์ข้อมูลในแต่ละfeature**
    """)
    st.write(cvd.describe())

    st.write("""
    **วิเคราะห์ว่าข้อมูลมีคนที่เป็นโรคและไม่เป็นโรคใกล้เคียงกันหรือไม่**
    """)
    st.write("0 is not diseased, 1 is diseased")
    st.write(cvd['cardio'].value_counts(normalize=True))

    st.write("กราฟแสดงการเป็นโรคและไม่เป็นโรคของคนที่อายุน้อยจนถึงอายุมาก")
    image3 = Image.open('./data/ageChart.png')
    st.image(image3, caption='age graph')

    st.write("กราฟแสดงการการกระจายตัวของแต่ละfeature")
    image4 = Image.open('./data/distributionPic.png')
    st.image(image4, caption='distribution graph')


    

    



