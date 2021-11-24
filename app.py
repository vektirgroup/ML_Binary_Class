import streamlit as st
import pandas as pd
import numpy as np

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




