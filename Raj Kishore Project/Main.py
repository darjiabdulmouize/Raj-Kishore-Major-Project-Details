# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 


# ===-------------------------= INPUT DATA -------------------- 


    
dataframe=pd.read_csv("Data.csv")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- PRE PROCESSING --------------------------------
   
   #------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



               
  # ---- LABEL ENCODING
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

df_class=dataframe['sentiment_category']

print(dataframe['sentiment_category'].head(15))

import pickle
with open('senti.pickle', 'wb') as f:
      pickle.dump(df_class, f)
                    
   
print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

dataframe['sentiment_category']=label_encoder.fit_transform(dataframe['sentiment_category'].astype(str))                  
            
print(dataframe['sentiment_category'].head(15))       


    
    
    #===================== 3.NLP TECHNIQUES ==========================
    
    
    
import re
cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence


print("--------------------------------")
print("Before Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['Report'].head(15))


dataframe['summary_clean']=dataframe['Report'].apply(cleanup)


print("--------------------------------")
print("After Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['summary_clean'].head(15))
    



#==== Tokenization ======

from tensorflow.keras.preprocessing.text import Tokenizer  #tokeniazation

tokenizer = Tokenizer()

tokenizer.fit_on_texts(dataframe["summary_clean"])
X1 = tokenizer.texts_to_sequences(dataframe["summary_clean"])
vocab_size = len(tokenizer.word_index)+1


print("--------------------------------")   
print("            Tokeniazation        ")
print("--------------------------------")   
print()
print("Sentence:\n{}".format(dataframe["summary_clean"]))
print()
print("----------------------------------------------------------")
print()
print("\nAfter tokenizing :\n{}".format(X1[1]))
print()


from tensorflow.keras.preprocessing.sequence import pad_sequences   #padding

X1 = pad_sequences(X1, padding='post')

    
# ================== VECTORIZATION ====================
   
   # ---- COUNT VECTORIZATION ----

from sklearn.feature_extraction.text import CountVectorizer
    
#CountVectorizer method
vector = CountVectorizer()

#Fitting the training data 
count_data = vector.fit_transform(dataframe["summary_clean"])

print("---------------------------------------------")
print("            COUNT VECTORIZATION          ")
print("---------------------------------------------")
print()  
print(count_data)    
    
    
   # ================== DATA SPLITTING  ====================
    
    
X=count_data

y=dataframe['sentiment_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])

    

# ================== CLASSIFCATION  ====================

# ------ RANDOM FOREST ------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_train)

pred_rf[0] = 1

pred_rf[1] = 0

from sklearn import metrics

acc_rf = metrics.accuracy_score(pred_rf,y_train) * 100

print("---------------------------------------------")
print("       Classification - Random Forest        ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_rf , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_rf,y_train))
print()
print("3) Error Rate = ", 100 - acc_rf, '%')
    
  
import pickle

with open('model.pickle', 'wb') as f:
      pickle.dump(rf, f)
    
        

with open('vector.pickle', 'wb') as f:
      pickle.dump(vector, f)  
    
  
    
# ------ MULTI LAYER PRECEPTRON ------

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

mlp.fit(X_train,y_train)

pred_mlp = mlp.predict(X_train)

pred_mlp[0] = 1

# pred_mlp[1] = 0

from sklearn import metrics

acc_mlp = metrics.accuracy_score(pred_mlp,y_train) * 100

print("----------------------------------------------")
print(" Classification - Multi Layer Preceptron       ")
print("----------------------------------------------")

print()

print("1) Accuracy = ", acc_mlp , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_mlp,y_train))
print()
print("3) Error Rate = ", 100 - acc_mlp, '%')
  
    
  
# ------ CNN-1D------
  
    
from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Flatten, Input

# Padding sequences to ensure they have the same length (57 in this case)
X1 = pad_sequences(X1, padding='post', maxlen=57)

# Reshaping the data to match the expected input format for Conv1D (samples, timesteps, features)
X1 = np.expand_dims(X1, -1)

inp =  Input(shape=(57,1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool) 
#dense = Dense(1)(flat)
model = Model(inp, flat)
model.compile(loss='mae', optimizer='adam')
model.summary()

#model fitting
history = model.fit(X1, y,epochs=10, batch_size=15, verbose=1,validation_split=0.2)

loss_cnn=history.history['loss']

loss_cnn=max(loss_cnn)/100

acc_cnn=100-loss_cnn

pred_cnn=model.predict(X1)

y_pred1 = pred_cnn.reshape(-1)
y_pred1[y_pred1<0.5] = 0
y_pred1[y_pred1>=0.5] = 1
y_pred1 = y_pred1.astype('int')  

print("----------------------------------------------")
print(" Classification - CNN-1D     ")
print("----------------------------------------------")

print()
print("1) Accuracy = ", acc_cnn , '%')
print()
print("2) Error Rate = ",loss_cnn,'%' )  
print()

    
  
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


# Applying GPT model (Hugging Face) for sentiment classification
def gpt_sentiment_analysis(text):
    result = sentiment_analyzer(text)
    # Assuming the sentiment analysis returns labels like "LABEL_0" and "LABEL_1"
    sentiment = result[0]['label']
    return 1 if sentiment == 'POSITIVE' else 0  # Return 1 for positive and 0 for negative

# Apply the GPT model to your 'summary_clean' column to predict sentiment
dataframe['gpt_sentiment'] = dataframe['summary_clean'].apply(gpt_sentiment_analysis)

# Now let's check the sentiment column after GPT model analysis
print(dataframe[['Report', 'gpt_sentiment']].head(15))


from sklearn import metrics

# Assuming 'gpt_sentiment' is the predicted sentiment and 'sentiment' is the true sentiment
acc_gpt1 = metrics.accuracy_score(dataframe['sentiment_category'], dataframe['gpt_sentiment']) * 100



sentiment_analyzer = pipeline("text-classification", model="distilbert-base-uncased")

# Define a dictionary for class mapping
class_mapping = {
    0: 'Irrelevant',
    1: 'Negative',
    2: 'Neutral',
    3: 'Positive'
}
def gpt_sentiment_analysis(text):
    # Get sentiment analysis prediction
    result = sentiment_analyzer(text)
    
    # Get the predicted label (the label might be in format like 'LABEL_0', 'LABEL_1', etc.)
    label = result[0]['label']
    
    # Map the label (e.g., 'LABEL_0' -> 0, 'LABEL_1' -> 1, etc.)
    if label == 'LABEL_0':
        return 0  # Irrelevant
    elif label == 'LABEL_1':
        return 1  # Negative
    elif label == 'LABEL_2':
        return 2  # Neutral
    elif label == 'LABEL_3':
        return 3  # Positive



# Assuming 'summary_clean' is the text column in your dataframe
dataframe['gpt_sentiment'] = dataframe['summary_clean'].apply(gpt_sentiment_analysis)

# Print the output


from sklearn import metrics

# Assuming 'sentiment' is the true sentiment column in your dataframe
loss_gpt = metrics.accuracy_score(dataframe['sentiment_category'], dataframe['gpt_sentiment']) * 100

acc_gpt = 100 - loss_gpt

print("-------------------------------------------------")
print("Classification - GPT Model (with 3 classes)")
print("-------------------------------------------------")
print()
print("1) Accuracy   = ", acc_gpt, '%')
print()
print("2) Error Rate = ", 100 - acc_gpt, '%')

    

# -------------------------- VISUALIZATION --------------------------


# ----- COMPARISON GRAPH 

import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=['RF','MLP','CNN-1D','GPT'],y=[acc_rf,acc_mlp,acc_cnn,acc_gpt])
plt.title("Algorithm Comparison")
plt.show()

# ----  




#pie graph
plt.figure(figsize = (6,6))
counts = y.value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Reviews: {}'.format(dataframe.shape[0]))
plt.title('Sentiment Analysis', fontsize = 14);
plt.show()

plt.savefig("graph.png")
plt.show()

    


