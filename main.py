# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:39:16 2022

@author: Gamal
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

st.title("Machine Learning Algorithms Demo")
st.write("""## Which Algorithm is the best
            "KNN" or "SVM" or "Random Forest"
""")

dataset_name = st.selectbox("Select Dataset",("Iris","Cancer","Wine","My Dataset"))

classifier_name = st.sidebar.selectbox("Select Algorithm",("KNN","SVM","Random Forest"))
#df = pd.read_csv("NN_trainingDataSet.csv")
#st.write("Shape of Dataset", df.shape)
#st.write("My Dataset", df.head())
def get_dataset (dataset_name): 
    if dataset_name == "Iris" :
        data = datasets.load_iris()
    elif dataset_name == "Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "My Dataset" :
       df = pd.read_csv("NN_trainingDataSet.csv")
       data = np.array(df)
        
    else:
        data = datasets.load_wine()
    
    st.write("Type of data",type(data))
    X = data.data
    y = data.target
    return X,y

X, y = get_dataset(dataset_name)
#st.write("Dataset Head", X.head())
st.write("Shape of Dataset", X.shape)

def Adjust_Parameters(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("Adjust K",1,15)
        params["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("Adjust C",0.01,10.0)
        params["C"] = C
    else :
        max_depth = st.sidebar.slider("Adjust max_depth",2,20)
        n_estimators = st.sidebar.slider("Adjust n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = Adjust_Parameters(classifier_name)   

def get_Algorithm(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif classifier_name == "SVM":
        clf = SVC(C = params["C"])
    else :
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                           max_depth = params["max_depth"],random_state=1234)
        
    return clf

clf = get_Algorithm(classifier_name, params)

######### Classification  ########################

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state = 1234)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
acc = accuracy_score(y_test, y_predict)
st.write(f"Algorithm name = {classifier_name}")
st.write(f"The Accuracy = {acc}")

########  Plot  ################################
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1] 

fig = plt.figure()  
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

## Show plot
st.pyplot(fig)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
        
