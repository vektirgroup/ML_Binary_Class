import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px

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

st.code('''
    # Initialize setup ============
    from pycaret.classification import *
    s = setup(data = data, target = 'default', session_id=123)
''')

# Compare models ============
st.write('''
         Now that we have the data and our project setup we will use
         compare model by running compare_models() function, it returns
         a highlighted table of models that it automatically comapred for 
         us .
         ''')
st.code('''
    # Compare Models ============
    best_model = compare_models()
''')
st.image('img/compare_models.png', caption='Pycaret -> compare_models()')

# Select model ============
st.write('''
         After reviewing the several different models we will choose the 
         ridge regression model to further explore 
         ''')
st.code('''
    # Create Models ============
    rdg_mdl = create_model('ridge')
''')
st.image('img/ridge_classifier_model.png', caption='Pycaret -> create_model("ridge")')

# Tune model ============
st.write('''
    After creating the model we can easily tune the model using tune_model() 
    ''')
st.code('''
    # Tune model
    tuned_rdg_mdl = tune_model(rdg_mdl)
''')
st.image('img/tuned.png', caption='Pycaret -> tune_model("model_name")')

# Predict model ============
st.write('''
         Once the model has been tuned to your liking we can use the predict function to 
         predict on the unseen data.
         ''')
st.code('''
    # Predict model    
    rdg_unseen_predictions = predict_model(final_rdg)
''')
st.image('img/predict model.png', caption='Pycaret -> predict_model("model_name")')


# Finalize model ============
st.write('''
         After creating the model we can easily tune the model using tune_model() 
         ''')
st.code('''
    #Finalize model    
    final_rdg = finalize_model(rdg_mdl_tune);
    print(final_rdg)
''')
st.image('img/finalize model.png', caption='Pycaret -> finalize_model("model_name")')


# Save model ============
st.write('''
         Finally, after creating, tuning, predicting and finalizing the model we will
         now save the model as a pickle file to be used later.
         ''')
st.code('''
    # Save Model     
    save_model(final_rdg,'final_reg_112521')
''')
st.image('img/saved_model.png', caption='Pycaret -> finalize_model("model_name")')
