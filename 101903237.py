# Name: Yogesh Sharma
""" If matplotlib,sklearn,streamlit is not install We can install them using commands as follows:
pip install matplotlib
pip install scikit-learn
pip install streamlit """

#here we import pandas to read csv files
import pandas as pd    

#we use numpy to simply our data
import numpy as np      

#matplotlib for ploting graph
import matplotlib.pyplot as plt    

#here we use streamlit framework to create our dashboard
import streamlit as st    

#using PIL we can import Images
from PIL import Image                
img = Image.open('.\icons\ml.png')
img1 = Image.open('.\icons\ds.png')

#Importing all Classifiers so that we can find accuracy
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Import PCA to convert all input variables into 2 input variables so that we can plot graph in 2D
from sklearn.decomposition import PCA

#here we set title of page and give icon to it
st.set_page_config(page_title="Data Science Dashboard",page_icon=img,layout="wide")
st.title("💻 Data Science Dashboard")

st.write("""
### Explore Different Classifiers
##### Here we can explore different classifiers using different datasets and find out Which one is best?
""")

#here subheader is created in sidebar
st.sidebar.subheader('Create/Filter your search')

#using selectbox we can select different choices
dataset_name = st.sidebar.selectbox("Select Dataset 📍", ("Wheat Dataset","BankNoteAuthentication Dataset",
                                    "Iris Dataset","Breast Cancer Dataset","Sonar Dataset",
                                    "Wine Quality Dataset", "Diabetes Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier📍", ("KNN","SVM", "Random Forest"))

#here we import all our datasets
def get_dataset(dataset_name):
    if dataset_name == "Wheat Dataset":
        df = pd.read_csv("datasets\Seed_Data.csv")
    elif dataset_name == "BankNoteAuthentication Dataset":
        df = pd.read_csv("datasets\BankNoteAuthentication.csv")
    elif dataset_name == "Iris Dataset":
        df = pd.read_csv("datasets\Iris.csv")
    elif dataset_name == "Breast Cancer Dataset":
        df = pd.read_csv("datasets\Breast_cancer.csv")
    elif dataset_name == "Sonar Dataset":
        df = pd.read_csv("datasets\sonar.csv")
    elif dataset_name == "Wine Quality Dataset":
        df = pd.read_csv("datasets\Wine.csv")
    else:
        df = pd.read_csv("datasets\Diabetes.csv")

    #using iloc we seprates all input variables in X and all output variables in y    
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    return X,y

#here we calling get_dataset function
X,y = get_dataset(dataset_name)

#X.shape is used to find number of rows and coloums of X
st.write("Shape of Dataset",X.shape)

#using numpy unique we are calculating number of different classes
st.write("Number of Classes",len(np.unique(y)))

#using sidebar slider user can give values of parameters that are used in classifiers
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params
#the parameters that are return by add_parameter function are stored in params
params=add_parameter_ui(classifier_name)

#get_classifer function is applying classification to selected dataset
def get_classifer(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                     max_depth=params["max_depth"], random_state=1234)

    return clf

clf = get_classifer(classifier_name,params)

#Applying Classification and convert our dataset into 75% for training and 25% for testing
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=1234)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

#here we first apply PCA to convert multi input variables to only 2 input variable so that we plot 2D graph
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

#Show plot
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.sidebar.subheader("""Created By Yogesh Sharma""")
st.sidebar.image('icons\ds.png', width = 200)