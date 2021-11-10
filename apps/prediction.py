import streamlit as st 
import numpy as np 
import datetime
from datetime import date

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def app():
    st.title('Cardiovascular disease prediction')

    st.write("""
    \*** Please fill out all fields for good prediction. \**\*
    """)

    #----------------------
    cvd = pd.read_csv('./data/cvd_dataset.csv', sep=';', index_col=["id"])
    #st.write(cvd.shape[0])
    #st.dataframe(cvd.head(9))
    cvd=cvd.drop(cvd[cvd['ap_lo']> cvd['ap_hi']].index)
    # st.write(cvd.shape[0])
    cvd=cvd.drop(cvd[(cvd['ap_hi'] > cvd['ap_hi'].quantile(0.975)) | (cvd['ap_hi'] < cvd['ap_hi'].quantile(0.025))].index)
    # st.write(cvd.shape[0])
    cvd=cvd.drop(cvd[(cvd['ap_lo'] > cvd['ap_lo'].quantile(0.975)) | (cvd['ap_lo'] < cvd['ap_lo'].quantile(0.025))].index)
    # st.write(cvd.shape[0])
    cvd['bmi'] = cvd['weight']/(cvd['height']/100)**2
    #st.dataframe(cvd.head(9))

    X = cvd.drop('cardio', axis=1)
    Y = cvd.cardio
    # st.dataframe(X.head(9))
    # st.dataframe(Y.head(9))

    #Preprocessing
    scaler = StandardScaler().fit(X)
    standard_X = scaler.transform(X)
    #st.write(standard_X)

    dfX=pd.DataFrame(standard_X)
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train_kfold = dfX.iloc[train_index]
        X_test_kfold = dfX.iloc[test_index]
        y_train_kfold = pd.Series(Y.iloc[train_index])
        y_test_kfold = pd.Series(Y.iloc[test_index])
        lgtr = LogisticRegression().fit(X_train_kfold, y_train_kfold)
    acc_lgtr = round(lgtr.score(X_train_kfold, y_train_kfold)*100, 2)
    #st.write(acc_lgtr)
    acc_test_lgtr = round(lgtr.score(X_test_kfold, y_test_kfold)*100, 2)
    #st.write(acc_test_lgtr)


    #-------------------
    gender = st.selectbox(
        'Gender',
        ('Male', 'Female')
    )

    birth = st.date_input(
        "Date of birth [Year/Month/Day]",
        datetime.date(1980, 1, 1),
        help="ปีค.ศ./เดือน/วัน"
    )

    height = st.number_input('Height(cm)', min_value=0, max_value=250)

    weight = st.number_input('Weight(kg)', min_value=0.0, max_value=300.0)

    sbp = st.slider(
         'Systolic blood pressure (mmHg)',
         min_value=0, max_value=250, value=120, help="ค่าความดันโลหิตตัวบน")

    dbp = st.slider(
         'Diastolic Blood Pressure (mmHg)',
         min_value=0, max_value=250, value=80, help="ค่าความดันโลหิตตัวล่าง")

    if (dbp>sbp):
        st.error("Systolic blood pressure ต้องมากกว่า Diastolic Blood Pressure")

    chol = st.selectbox(
        'Cholesterol',
        ('normal', 'above normal', 'well above normal')
    )

    glu = st.selectbox(
        'Glucose',
        ('normal', 'above normal', 'well above normal')
    )

    smoke = st.checkbox('ปัจจุบันสูบบุหรี่')

    alco = st.checkbox('ปัจจุบันมีการดื่มแอลกอฮอล์')

    act = st.checkbox('มีการออกกำลังกายเป็นประจำ')

    bmi=0.00
    if (height > 0):
        bmi = weight/(height/100)**2;

    #st.write(birth, gender, height, weight, sbp, dbp, chol, glu, smoke, alco, act, bmi)

    obese="เกณฑ์"
    if(bmi<18.5):
        obese="คุณอยู่ในเกณฑ์น้ำหนักน้อย"
    elif(bmi>22.9):
        obese="คุณอยู่ในเกณฑ์อ้วน"
    else:
        obese="คุณอยู่ในเกณฑ์น้ำหนักปกติ"


    def tonum(bool):
        if(bool):
            return 1
        else:
            return 0
    
    def texttonum(str):
        if(str=="normal"):
            return 1
        elif(str=="above normal"):
            return 2
        else:
            return 3


    if st.button('Predict'):

        if(height==0):
            st.error("กรุณากรอกข้อมูลส่วนสูง")
        if(weight==0):
            st.error("กรุณากรอกข้อมูลน้ำหนัก")

        if (height!=0) | (weight!=0):

            if(gender=="Male"):
                gd=2
            elif(gender=="Female"):
                gd=1

            td=[int(date.today().strftime("%Y")), int(date.today().strftime("%m")), int(date.today().strftime("%d"))]
            #st.write(td)
            tdsec=datetime.datetime(td[0],td[1],td[2]).timestamp()
             #st.write(tdsec)
            bd=[int(birth.strftime("%Y")), int(birth.strftime("%m")), int(birth.strftime("%d"))]
            #st.write(bd)
            bdsec=datetime.datetime(bd[0],bd[1],bd[2]).timestamp()
            #st.write(bdsec)
            period=(tdsec-bdsec)/86400
            #st.write(period)

            smoke_b=tonum(smoke)
            alco_b=tonum(alco)
            act_b=tonum(act)

            chol_n=texttonum(chol)
            glu_n=texttonum(glu)

            X_pred=np.array([period, gd, height, weight, sbp, dbp, chol_n, glu_n, smoke_b, alco_b, act_b, bmi])
            #st.write(X_pred)
            X_reshape=X_pred.reshape(1, -1)
            std_X = scaler.transform(X_reshape)
            #st.write(std_X)

            result=lgtr.predict(std_X)
            #st.write(result)

            if(result==0):
                prediction="____คุณไม่เป็นโรค____ :)"
            elif(result==1):
                prediction="____คุณเป็นโรค____ :("

            st.write("---------------"*6)
            st.write("BMI: ",round(bmi,2))
            st.write(obese)
            st.code(prediction)
            st.write("---------------"*6)



            

