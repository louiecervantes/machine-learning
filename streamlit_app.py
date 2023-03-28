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

# Define the Streamlit app
def app():
    st.header("Welcome to Machine Learning")
    st.subheader("Louie F. Cervantes M.Eng. \n(c) 2023 WVSU College of ICT")
    
    st.title("This app will compare machine learning algorithms")
    st.write('We use a dataset of mobile phone attributes to train the ML classifiers. We then split the data into training and test sets. Each classifier is tasked to predict the pricing class then the accuracy of each one is measured by counting the number of correct predictions over the entire test set.')

    # Load the mobile phone dataset
    df=pd.read_csv('mobilephones.csv', header=0)
    st.dataframe(df, width=800, height=400)
    
    if st.button('Display Info'):
        fig, ax = plt.subplots(figsize=(12,12))
        sns.heatmap(df1, annot=True,cmap = "Blues", fmt= '.0f',
        ax=ax,linewidths = 5, cbar = False,
        annot_kws={"size": 16})
        plt.xticks(size = 18)
        plt.yticks(size = 12, rotation = 0)
        plt.ylabel("Variables")
        plt.title("Descriptive Statistics", size = 16)
        st.pyplot(fig)
     

# Run the app
if __name__ == "__main__":
    app()
