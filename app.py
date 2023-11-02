import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title(' MEDICAL DIAGNOSTIC WEB APP ⚕ ')
st.subheader('Does the patient has diabetics?')
df=pd.read_csv(r'diabetes.csv')

if st.sidebar.checkbox('View Data', False):
    st.write(df)
#if st.sidebar.checkbox('View Distributions', False):
    #df.hist()
   # plt.tight_layout()
    #st.pyplot()
    
# step 1 : load the pickled model
model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()

# step 2. get the front end user input 

pregs=st.number_input('Pregnancies',0,20,0)
plas=st.slider('Glucose',40,200,40)
pres=st.slider('BloodPressure',22,150,20)
skin=st.slider('SkinThickness',7,99,7)
insulin=st.slider('Insulin',14,850,14)
bmi=st.slider('BMI',18,70,18)
dpf=st.slider('DiabetesPedigreeFunction',0.05,2.50,0.05)
age=st.slider('Age',21,90,21)

#step 3: get the model input (convert user input to model impout)
input_data=[[pregs,plas,pres,skin,insulin,bmi,dpf,age]]

#step 4: get the prediction and print the results 

prediction=clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader('Non Diabetic')
    else:
        st.subheader('Diabetic')
