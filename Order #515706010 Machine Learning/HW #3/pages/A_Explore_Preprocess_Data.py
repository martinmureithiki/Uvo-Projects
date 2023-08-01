import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from helper_functions import fetch_dataset, clean_data, summarize_review_data, display_review_keyword, remove_review

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 3 - Predicting Product Review Sentiment Using Classification")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################

# Checkpoint 1
def remove_punctuation(df, features):
    """
    This function removes punctuation from features (i.e., product reviews)

    Input: 
        - df: the pandas dataframe
        - feature: the features to remove punctation
    Output: 
        - df: dataframe with updated feature with removed punctuation
    """
    # check if the feature contains string or not
    translator = str.maketrans('', '', string.punctuation)

    # applying translate method eliminating punctuations
    for feature in features:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].apply(lambda x: x.translate(translator))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Confirmation statement
    st.write('Punctuation was removed from {}'.format(features))
    return df

# Checkpoint 2
def word_count_encoder(df, feature, word_encoder):
    """
    This function performs word count encoding on feature in the dataframe

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform word count encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with word count feature
    """
    
    # if 'Word Count' in word_encoder:
    # create a CountVectorizer object
    count_vect = CountVectorizer()

    # fit and transform the feature using the CountVectorizer
    freq_counts = count_vect.fit_transform(df[feature])

    # convert the resulting sparse matrix to a dense array and then to a dataframe
    freq_counts_df = pd.DataFrame(freq_counts.toarray())

    # add prefix to column names
    freq_counts_df = freq_counts_df.add_prefix('word_count_')

    # concatenate the resulting word count dataframe to the original dataframe
    df = pd.concat([df, freq_counts_df], axis=1)

    # (Uncomment code) Show confirmation statement
    st.write('Feature {} has been word count encoded from {} reviews.'.format(feature, len(df)))
    # st.write(word_count_df.head())

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Save variables for restoring state
    word_encoder.append('Word Count')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect

    return df

# Checkpoint 3
def tf_idf_encoder(df, feature, word_encoder):
    """
    This function performs tf-idf encoding on the given features

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform tf-idf encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with tf-idf encoded feature
    """
    # for feature in features:
    # if 'TF-IDF' in word_encoder:
    # create a CountVectorizer object
    count_vect = CountVectorizer()

    # fit and transform the feature using the CountVectorizer
    freq_counts = count_vect.fit_transform(df[feature])

    # create a TfidfTransformer object
    tfidf_transformer = TfidfTransformer()

    # fit and transform the frequency counts using the TfidfTransformer
    tfidf_features = tfidf_transformer.fit_transform(freq_counts)

    # convert the resulting sparse matrix to a dense array and then to a dataframe
    tfidf_features_df = pd.DataFrame(tfidf_features.toarray())

    # add prefix to column names 
    tfidf_features_df = tfidf_features_df.add_prefix('tf_idf_word_count_')

    # concatenate the resulting tf-idf dataframe to the original dataframe
    df = pd.concat([df, tfidf_features_df], axis=1)

    # (Uncomment code) Show confirmation statement
    st.write('Feature {} has been TF-IDF encoded from {} reviews.'.format(feature, len(df)))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Save variables for restoring state
    word_encoder.append('TF-IDF')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect
    st.session_state['tfidf_transformer'] = tfidf_transformer
    return df

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the Amazon Product Reviews dataset. Millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. See the unprocesses dataset below.')

    st.dataframe(df)

    # Remove irrelevant features
    df, data_cleaned = clean_data(df)
    if (data_cleaned):
        st.markdown('The dataset has been cleaned. Your welcome!')

    ############## Task 1: Remove Punctation
    st.markdown('### Remove punctuation from features')
    removed_p_features = st.multiselect(
        'Select features to remove punctuation',
        df.columns,
    )
    if (removed_p_features):
        df = remove_punctuation(df, removed_p_features)
        # Display updated dataframe
        st.dataframe(df)
        st.write('Punctuation was removed from {}'.format(removed_p_features))

    # Summarize reviews
    st.markdown('### Summarize Reviews')
    object_columns = df.select_dtypes(include=['object']).columns
    summarize_reviews = st.selectbox(
        'Select the reviews from the dataset',
        object_columns,
    )
    if(summarize_reviews):
        # Show summary of reviews
        summary = summarize_review_data(df, summarize_reviews)

    # Inspect Reviews
    st.markdown('### Inspect Reviews')

    review_keyword = st.text_input(
        "Enter a keyword to search in reviews",
        key="review_keyword",
    )

    # Display dataset
    st.dataframe(df)

    if (review_keyword):
        displaying_review = display_review_keyword(df, review_keyword)
        st.write(displaying_review)

    # Remove Reviews: number_input for index of review to remove
    st.markdown('### Remove Irrelevant/Useless Reviews')
    review_idx = st.number_input(
        label='Enter review index',
        min_value=0,
        max_value=len(df),
        value=0,
        step=1)

    if (review_idx):
        df = remove_review(df, review_idx)
        st.write('Review at index {} has been removed'.format(review_idx))

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)
    word_encoder = []

    word_count_col, tf_idf_col = st.columns(2)

    ############## Task 2: Perform Word Count Encoding
    with (word_count_col):
        text_feature_select_int = st.selectbox(
            'Select text features for encoding word count',
            string_columns,
        )
        if (text_feature_select_int and st.button('Word Count Encoder')):
            df = word_count_encoder(df, text_feature_select_int, word_encoder)

    ############## Task 3: Perform TF-IDF Encoding
    with (tf_idf_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for encoding TF-IDF',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('TF-IDF Encoder')):
            df = tf_idf_encoder(df, text_feature_select_onehot, word_encoder)

    # Show updated dataset
    if (text_feature_select_int or text_feature_select_onehot):
        st.write(df)

    # Save dataset in session_state
    st.session_state['data'] = df

    st.write('Continue to Train Model')
