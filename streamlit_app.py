#Write a simple app that reads the user input and display the output
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import linear_model
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Define the Streamlit app
def app():
    st.header("Welcome to Machine Learning")
    st.subheader("Louie F. Cervantes M.Eng. \n(c) 2023 WVSU College of ICT")
    
    st.title("This app will compare machine learning algorithms")
    st.write('We use a dataset of mobile phone attributes to train the ML classifiers. We then split the data into training and test sets. Each classifier is tasked to predict the pricing class then the accuracy of each one is measured by counting the number of correct predictions over the entire test set.')

    # Load the mobile phone dataset
    df=pd.read_csv('mobilephones.csv', header=0)
    st.dataframe(df, width=800, height=400)
    desc = df.describe().T
    df1 = pd.DataFrame(index=['battery_power', 'blue', 'clock_speed', 'dual_sim',
                          'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 
                          'n_cores', 'pc', 'px_height', 'px_width', 'ram', 
                          'sc_h', 'sc_w', 'talk_time', 'three_g','touch_screen',
                          'wifi', 'price_range'], 
                   columns= ["count","mean","std","min",
                             "25%","50%","75%","max"], data= desc )
    if st.button('Begin'):
        fig, ax = plt.subplots(figsize=(12,12))
        sns.heatmap(df1, annot=True,cmap = "Blues", fmt= '.0f',
        ax=ax,linewidths = 5, cbar = False,
        annot_kws={"size": 16})
        plt.xticks(size = 18)
        plt.yticks(size = 12, rotation = 0)
        plt.ylabel("Variables")
        plt.title("Descriptive Statistics", size = 16)
        st.pyplot(fig)
        st.write("The number of unique class for each attribute")
        st.write(df.nunique())
        st.write("More useful information about the dataset")
        st.write(df.describe().T) 

    #The default test size is 20%
    test_size = 0.2
    st.write('Set the percentage of the test set.')
    options = ['10%', '20%', '30%']
    selected_option = st.selectbox('Select the proportion of the test set', options)
    if selected_option=='10%':
        test_size = 0.1
    if selected_option=='20%':
        test_size = 0.2
    if selected_option=='30%':
         test_size = 0.3
        
    #load the data and the labels into training and test sets
    X = df.values[:,0:-1]
    y = df.values[:,-1]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
    st.write('The k-nearest neighbor (k-NN) classifier is a type of machine learning algorithm that can be used for both classification and regression problems.In the k-NN classifier, the training data is used to determine the k closest neighbors to a new input data point. The k closest neighbors are determined based on a distance metric, such as Euclidean distance or Manhattan distance. The class of the new data point is then predicted based on the class of the majority of its k-nearest neighbors. For example, if k = 3 and two of the neighbors belong to class A while one belongs to class B, the new data point is classified as belonging to class A.')
    if st.button('Run KNN'):
        # Number of nearest neighbors
        num_neighbors = 5
        # Create a K Nearest Neighbors classifier model
        clfKNN = neighbors.KNeighborsClassifier(num_neighbors,
        weights='distance')
        clfKNN.fit(X_train, y_train)
        y_test_pred = clfKNN.predict(X_test)
        st.subheader("KNN classification performance")
        st.write(classification_report(y_test, clfKNN.predict(X_test)))
        cmKNN = confusion_matrix(y_test, y_test_pred)
        st.write("The confusion matrix")
        st.write(cmKNN)
  
    st.write('Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification, regression and outlier detection. The main idea behind SVM is to find the hyperplane that best separates the different classes of data. In a binary classification problem, the hyperplane is a line that maximizes the margin between the two classes. The margin is defined as the distance between the hyperplane and the closest data points from each class. The hyperplane that maximizes the margin is also known as the maximum-margin hyperplane. The SVM algorithm can be used for linearly separable and non-linearly separable data by mapping the input data into a higher-dimensional feature space where it is more likely to be linearly separable. This is done by using a kernel function that computes the similarity between two data points in the higher-dimensional feature space.')
    if st.button('Run SVM'):
        clfSVM = svm.SVC(kernel='linear', C=1000)
        clfSVM.fit(X_train, y_train)
        y_test_pred = clfSVM.predict(X_test)
        st.subheader("SVM Classification Performance")
        st.write(classification_report(y_test, clfSVM.predict(X_test)))        
        cmSVM = confusion_matrix(y_test, y_test_pred)
        st.write("The confusion matrix")
        st.write(cmSVM)

    st.write('Extreme Random Forest (or Extra Trees) is an ensemble learning method used for classification, regression and feature selection tasks. It is a variant of the Random Forest algorithm and works by constructing multiple decision trees using random subsets of the training data and random subsets of the features. The difference between Random Forest and Extra Trees lies in the way the decision trees are constructed. In Random Forest, each decision tree is constructed by randomly selecting a subset of the features and finding the best split among those features. In Extra Trees, each decision tree is constructed using random splits, regardless of the quality of the split.')
    if st.button('Run ERF'):
        clfERF = ExtraTreesClassifier( n_estimators=100, max_depth=5, random_state=0)
        clfERF.fit(X_train, y_train)
        y_test_pred = clfERF.predict(X_test)
        st.subheader("ERF Classification Performance")
        st.write(classification_report(y_test, clfERF.predict(X_test)))        
        cmERF = confusion_matrix(y_test, y_test_pred)
        st.write("The confusion matrix")
        st.write(cmERF)


# Run the app
if __name__ == "__main__":
    app()
