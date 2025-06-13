import pickle
import numpy as np
import streamlit as st
from sklearn.svm import SVC

st.title("Crop Reccomendation")



N = st.number_input("Enter the ratio of Nitrogen content in soil")

P = st.number_input("Enter the ratio of Phosphorous content in soil")

K = st.number_input("Enter the ratio of Potassium content in soil")

temperature = st.number_input("Enter the temperature in degree Celsius")

humidity  = st.number_input("Enter the relative humidity in %")

ph         = st.number_input("Enter the ph value of the soil")

rainfall = st.number_input ("Enter the rainfall in mm")


inp = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

# reshape_inp = inp.reshape(-1,1)

if st.button("predict"):

    loaded_model = pickle.load(open('svm_model.sav', 'rb'))

    res = loaded_model.predict(inp)

    st.success(res)



