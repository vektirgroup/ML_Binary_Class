import streamlit as st
import pandas as pd
import numpy as np

# Get Data ==============
from pycaret.datasets import get_data
dataset = get_data('credit')

# Page Config ==============
st.set_page_config(
    page_title=None, 
    page_icon=None, 
    layout="wide", 
    initial_sidebar_state="auto", 
    menu_items={
         'About': "# This is a header. This is an *extremely* cool app!"
     })

# Sidebar section ==============
st.sidebar.title('Vektir Labs')
st.sidebar.header('Pycaret Tutorial')
st.sidebar.write('''
                 This labs is an example of the Machine Learning 
                 Binary Classification Algorithm using Pycaret!               
                 ''')
with st.sidebar.expander('Reference Links'):
    st.markdown('Pycaret')
    st.markdown('  - [Github](https://github.com/pycaret/pycaret)')
    st.markdown('  - [Website](https://pycaret.org/)')
    st.markdown('  - [Article](https://moez-62905.medium.com/introduction-to-binary-classification-with-pycaret-a37b3e89ad8d)')

# Main image ============
st.image('img/header.png',caption='')

# Main page ============
st.header('Binary Classification Lab')

# Intro ============
st.subheader('Introduction')
st.write('''
         Binary classification is a supervised machine learning technique where the goal 
         is to predict categorical class labels which are discrete and unordered such as 
         Pass/Fail, Positive/Negative, Default/Not-Default, etc. A few real-world use 
         cases for classification are listed below:
         
        - Medical testing to determine if a patient has a certain disease or not — the 
          classification property is the presence of the disease.
        
        - A “pass or fail” test method or quality control in factories, i.e. deciding if 
          a specification has or has not been met — a go/no-go classification.
        
        - Information retrieval, namely deciding whether a page or an article should be 
          in the result set of a search or not — the classification property is the relevance 
          of the article or the usefulness to the user.
         ''')

# Get Data & Preview  ============
st.subheader('Get Data')
st.write('''
         For this example we will be using one of Pycaret's build in datasets. 
         ''')
st.code('''
# Get example credit data 
from pycaret.datasets import get_data
dataset = get_data('credit')
        ''')
# Show the dataframe head ============
st.dataframe(dataset.head())

# Sample 5% of data to be used as unseen data ============
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# print the revised shape ============
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

st.write('''
         In order to demonstrate the use of the predict_model function on unseen data, 
         a sample of 1200 records (~5%) has been withheld from the original dataset to 
         be used for predictions at the end. This should not be confused with a train-test-split, 
         as this particular split is performed to simulate a real-life scenario. Another way 
         to think about this is that these 1200 customers are not available at the time of training 
         of machine learning models.
         ''')

st.code('''
# Sample 5% of data to be used as unseen data
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# Print the revised shape
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# Data for Modeling: (22800, 24)
# Unseen Data For Predictions: (1200, 24)        
''')

st.write('The next step is to initialize our setup function')

# Initialize setup ============
from pycaret.classification import *
# s = setup(data = data, target = 'default', session_id=123)

st.code('''
# Initialize setup ============
from pycaret.classification import *
s = setup(data = data, target = 'default', session_id=123)
''')