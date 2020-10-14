import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier

st.write("""
An ML Web Application for Iris Flower Prediction
"""
         )

st.sidebar.header('Parameters')


def input_iris_parameters():
    sepal_length = st.sidebar.slider('SEPAL LENGTH', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('SEPAL WIDTH', 4.3, 7.9, 5.4)
    petal_length = st.sidebar.slider('PETAL LENGTH', 4.3, 7.9, 5.4)
    petal_width = st.sidebar.slider('PETAL WIDTH', 4.3, 7.9, 5.4)
    user_input = {'SEPAL LENGTH': sepal_length,
                  'SEPAL WIDTH': sepal_width,
                  'PETAL LENGTH': petal_length,
                  'PETAL WIDTH': petal_width}
    user_input_feature = pd.DataFrame(user_input, index=[0])
    return user_input_feature

user_input_df=input_iris_parameters()

st.subheader('User Input Feature Selection')
st.write(user_input_df)

iris_data=datasets.load_iris()
X,Y= iris_data.data, iris_data.target

clf_gradientboost = GradientBoostingClassifier()
clf_randomforest = RandomForestClassifier()
clf_adaboost = AdaBoostClassifier()

clf_gradientboost.fit(X,Y)
clf_randomforest.fit(X,Y)
clf_adaboost.fit(X,Y)

prediction= {'Gradient Boost': iris_data.target_names[clf_gradientboost.predict(user_input_df)],
             'Random Forest': iris_data.target_names[clf_gradientboost.predict(user_input_df)],
              'Ada Boost': iris_data.target_names[clf_gradientboost.predict(user_input_df)]}

prediction_df = pd.DataFrame(prediction, index=[0])

st.subheader('Target classes and their labels')
st.write(iris_data.target_names)

st.subheader('Classifier Predictions')
st.write(prediction_df)





