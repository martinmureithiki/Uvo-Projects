import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import string
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.title('Deploy Application')

#############################################
enc = OrdinalEncoder()

# Checkpoint 12
def deploy_model(text):
    """
    Restore the trained model from st.session_state[‘deploy_model’] 
                and use it to predict the sentiment of the input data    
    Input: 
        - df: pandas dataframe with trained regression model
    Output: 
        - product_sentiment: product sentiment class, +1 or -1
    """

    try:
        # Restore trained model
        model = st.session_state['deploy_model']

        # Make predictions on test data
        y_pred = model.predict(text)

        # Convert predictions to +1 or -1 sentiment
        Product_sentiment = [1 if pred == 1 else -1 for pred in y_pred]

        return Product_sentiment

    except:
        st.error('Error: Could not deploy model. Please train the model first.')
  

###################### FETCH DATASET #######################
df = None
if 'data' in st.session_state:
    df = st.session_state['data']
else:
    st.write('### The Product Review Sentiment Application is under construction. Coming to you soon.')

# Deploy App!
if df is not None:
    #df.dropna(inplace=True)
    st.markdown('### Introducing the ML Powered Review Application')

    # Perform error checking for strings, chars, etc (garbage)

    # Input review
    st.markdown('### Use a trained classification method to automatically predict positive and negative reviews')
    
    user_input = st.text_input(
        "Enter a review",
        key="user_review",
    )
    if (user_input):
        st.write(user_input)

        translator = str.maketrans('', '', string.punctuation)
        # check if the feature contains string or not
        user_input_updates = user_input.translate(translator)
        
        if 'count_vect' in st.session_state:
            count_vect = st.session_state['count_vect']
            text_count = count_vect.transform([user_input_updates])
            # Initialize encoded_user_input with text_count as default
            encoded_user_input = text_count
            if 'tfidf_transformer' in st.session_state:
                tfidf_transformer = st.session_state['tfidf_transformer']
                encoded_user_input = tfidf_transformer.transform(text_count)
            
            #product_sentiment = st.session_state["deploy_model"].predict(encoded_user_input)
            product_sentiment = deploy_model(encoded_user_input)
            if(product_sentiment):
                st.write('The product has a positive sentiment')
            else:
                st.write('The product has a negative sentiment')