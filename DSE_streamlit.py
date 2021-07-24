import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.title("Welcome to DSE Exam Predictions")

st.header("Streamlit App by Student Name")

from PIL import Image
st.image(
    "https://static.reuters.com/resources/r/?m=02&d=20200424&t=2&i=1516278229&r=LYNXNPEG3N08G&w=640",
    width=400, # Manually Adjust the width of the image as per requirement
)

st.subheader("Data")

st.write("Data provided by sekrit organization")

st.subheader("DataFrame")

df = pd.read_csv("./DSE2021_cleaned.csv", index_col=False)
courses = df.columns.tolist()

showplots = st.checkbox('Show Plots')
if showplots:
  for c in courses:
    fig, ax = plt.subplots()
    ax = sns.histplot(data=df[c])
    st.pyplot(fig)

st.sidebar.title("Filter data")
option = st.sidebar.selectbox("Select Elective to Predict", courses[4:])

display = courses[:4]
display.append(option)
new_df = df[display]
new_df = new_df.dropna(subset=[option])
new_df

st.subheader("Define X and Y for Training")
#define X and y for model training

X=new_df.iloc[:,0:4]
y=new_df.iloc[:,-1:]

st.write("X data")
X
st.write("y target")
y

#import train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=9)
clf.fit(X_train,y_train)

features = X_train.columns.tolist()
features_score = clf.feature_importances_.tolist()
feature_output = zip(features, features_score)
st.write(f"These are the most important features for {option}:", list(feature_output ))

st.write("Max depth of this Decision tree is", clf.tree_.max_depth)

st.subheader("Model Predictions")

y_pred = clf.predict(X_test)
result_df = pd.DataFrame(X_test)
result_df[f"Predictions for {option}"] = y_pred
result_df

st.subheader("User Scores")
Chinese = st.sidebar.number_input('What is your Chinese Score?', step=1, min_value=1, max_value=7)
English = st.sidebar.number_input('What is your English Score?', step=1, min_value=1, max_value=7)
Math = st.sidebar.number_input('What is your Math Score?', step=1, min_value=1, max_value=7)
LS = st.sidebar.number_input('What is your LS Score?', step=1, min_value=1, max_value=7)

prediction = clf.predict(np.array([Chinese,English,Math,LS]).reshape(1, -1))
st.subheader("User Predictions")
user_df = pd.DataFrame({"Chinese":[Chinese],"English":[English],"Math":[Math],"LS":[LS]})
user_df[f"Predictions for {option}"] = prediction
user_df




