import pandas as pd
import numpy as np
import joblib
import streamlit as st

classifier = joblib.load('titanic_model.pkl')

def predict_survived(pclass,sex,age,sibsp,parch,fare,embarked):

    prediction = classifier.predict(pd.DataFrame({'pclass':[pclass],'sex':[sex],'age':[age],'sibsp':[sibsp],'parch':[parch],'fare':[fare],'embarked':[embarked]}))

    label = ['unsurvived','survived']

    return label[prediction[0]]


def main():
    st.title('Titanic prediction')
    html_temp="""
                <div style="background-color:red">
                <h2 style="color:white;text-align:center;">this our streamlit Deployment App</h2>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)

    pclass = st.text_input('pclass','tell me your passenger class')
    sex = st.radio('pick your gender',['male', 'female'])
    age = st.text_input('age','put your age')
    sibsp = st.text_input('sibsp','put your sibsp')
    parch = st.text_input('parch','put your parch')
    fare = st.text_input('fare','put your fare')
    embarked = st.radio('pick your embarked',['S', 'C', 'Q'])

    result =""

    if st.button('predict'):
        result = predict_survived(pclass,sex,age,sibsp,parch,fare,embarked)
    st.success('this person is {}'.format(result))


if __name__ =='__main__':
    main()




