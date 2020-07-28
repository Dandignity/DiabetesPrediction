import streamlit as st
import pandas as pd
#import matpplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
import seaborn as sns

##### Title\\\
st.write("""
	# Diabets Detections
	Detect if someone has diabetes using maching learning!
	""")

#Get Data
df = pd.read_csv("diabetes.csv")

st.subheader("Data Information")
st.dataframe(df)

## show statistics
st.write(df.describe())
#chart =  st.bar_chart(df)

if st.checkbox("	Chart"):
	all_columns = df.columns.to_list()
	feat_choices = st.multiselect("Choose a Feature",all_columns)
	new_df = df[feat_choices]
	st.line_chart(new_df)


#split data
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state = 1)

#Get the feature input from user
#st.subheader("prediction Value")
def get_user_input():
	Pregnancies = st.sidebar.number_input("Pregnancies", 0, 17, 3)
	Glucose = st.sidebar.number_input('Glucose', 0, 199, 117)
	Blood_Pressure 	= st.sidebar.number_input('Blood_Pressure', 0, 122, 72)
	Skin_Thickness	= st.sidebar.slider('Skin_Thickness', 0, 99, 23)
	Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.5)
	BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
	DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3)
	Age = st.sidebar.slider('Age', 0, 90, 30)


#store a  Dictionary varaibles
	user_data = {'Pregnancies': Pregnancies,
				'Glucose': Glucose,
				'Blood_Pressure': Blood_Pressure,
				'Skin_Thickness': Skin_Thickness,
				'Insulin': Insulin,
				'BMI': BMI,
				'DPF':DPF,
				'Age':Age
			}
#Transform the Data frame
	features = pd.DataFrame(user_data, index =[0])
	return features


#store the user input into variable
user_input = get_user_input()

#set a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

#create and train a model


st.subheader('Random Foreest:')
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)


st.subheader('Model accuracy Level')
#st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification: ')

st.write(prediction)
prediction_label = {"Positive":1,"Negative":0}
#final_result = get_key(prediction,prediction_label)
if prediction == 1:
	st.warning("This Patient is Diabetes Positive")
	
else:
	st.success("This Patient is Diabetes Negative")
	
st.subheader('KNN:')

KNeighborsClassifier = KNeighborsClassifier()
KNeighborsClassifier.fit(X_train,Y_train)


st.subheader('Model accuracy Level')
st.write(str(accuracy_score(Y_test,KNeighborsClassifier.predict(X_test))*100)+'%')

prediction = KNeighborsClassifier.predict(user_input)
pred_prob = KNeighborsClassifier.predict_proba(user_input)


st.subheader('Prediction: ')
#st.write(pred_prob)

#st.write(prediction)
prediction_label = {"Positive":1,"Negative":0}
if prediction == 1:
	st.warning("This Patient is Diabetes Positive")
	pred_probability_score = {"Negative	":pred_prob[0][0]*100,"Positive":pred_prob[0][1]*100}
	st.subheader("Prediction Probability Score using KNN")
	st.json(pred_probability_score)
else:
	st.success("This Patient is Diabetes Negative")
	pred_probability_score = {"Negative	":pred_prob[0][0]*100,"Positive":pred_prob[0][1]*100}
	st.subheader("Prediction Probability Score using KNN")
	st.json(pred_probability_score)

st.subheader('Logistics:') 	

LogisticRegression = LogisticRegression()
LogisticRegression.fit(X_train,Y_train)
prediction = LogisticRegression.predict(user_input)

#Class Model
st.subheader('Classification:')
st.write(prediction)
prediction_label = {"Positive":1,"Negative":0}
pred_prob = LogisticRegression.predict_proba(user_input)

#Predict Model
st.subheader('Prediction: ')
prediction_label = {"Positive":1,"Negative":0}
if prediction == 1:
	st.warning("This Patient is Diabetes Positive")
	pred_probability_score = {"Negative	":pred_prob[0][0]*100,"Positive":pred_prob[0][1]*100}
	st.subheader("Prediction Probability Score using Logistics")
	st.json(pred_probability_score)
else:
	st.success("This Patient is Diabetes Negative")
	pred_probability_score = {"Negative	":pred_prob[0][0]*100,"Positive":pred_prob[0][1]*100}
	st.subheader("Prediction Probability Score using Logistics")
	st.json(pred_probability_score)

	