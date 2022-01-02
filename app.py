import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

 

start = '2010-01-01'
end = '2021-12-1'
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#Describing Data
st.subheader('Data from 2010 - 2021')
st.write(df.describe())

#Visualisation
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart With 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart With 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g')
plt.plot(df.Close,'b') 
plt.plot(ma200,'r') 
st.pyplot(fig)


#splitting data into training and testing        ****************here we have not used tarin test and split
data_train= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])  #ie 70% of our data is used for training purpose
data_test= pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df) )])  # Here traing values are data starting from 70% value and it will go till end of data

from sklearn.preprocessing import MinMaxScaler #Coz we need our data between 0 and 1 in LSTM Model 
scaler = MinMaxScaler(feature_range =(0,1))

data_train_array = scaler.fit_transform(data_train)  #Our data is automatically converted into array as we used MinMaXScler funaction 

x_train = []
y_train = []
for i in range(100,data_train_array.shape[0]):  
  x_train.append(data_train_array[i-100 : i])       #This will append from 101- till end , then 102 - till end , then 103 - till end
  y_train.append(data_train_array[i,0])             #This will store values like 101 then 102 then 103............

# we will convert x_train and y_train into np.array so that we can feed our data to LSTM model
x_train , y_train = np.array(x_train) , np.array(y_train)

# Creating The ML Model
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50 , activation = 'relu', return_sequences= True, input_shape= (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60 , activation = 'relu', return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80 , activation = 'relu', return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120 , activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))      #Last is only one unit because we want single output that is Closing Price

# Compile the model
model.compile(optimizer= 'adam' , loss = 'mean_squared_error')

#Training the model
model.fit(x_train,y_train,epochs=10)  #actual code has 50 epoch  here we did only 10 for saving our time


 
past_100_days = data_train.tail(100)  #yeh last wala training data hai jo aapne ko aapne ko testing data k liye previous 100days data provide karega 
# iske aage aapn testing data jod denge taki aapne starting k 100 values waste na 100 ma banane mai
final_df = past_100_days.append(data_test,ignore_index = True)

#again we have scale this data between 0 and 1
input_data= scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)

#Making Predictions
y_predicted = model.predict(x_test)

scale_factor = 1/0.00770698
y_predicted = y_predicted  * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions Vs Orignal')
fig2 = plt.figure(figsize=(12,6))
plt.title('Accuracy')
plt.plot(y_test,'b',label = 'Orignal Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
