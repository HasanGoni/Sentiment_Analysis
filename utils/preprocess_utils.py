from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing as pr


from sklearn.model_selection import train_test_split

stemmer = PorterStemmer()
random_number = 42


def review_to_words(review):
    "Convert raw review string to sequence of words"
    # no need for this task. just for general task is used
    text = BeautifulSoup(review, 'html5lib').get_text()
    # lowering the words and removing other than alphanumeric word
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    words = [i for i in text.split() if i not in stopwords.words('english')]
    words = [PorterStemmer().stem(w) for w in words]

    return words


def preprocess_y_data(df):
    """Preprocess the dependent variable

    Args:
        df ([type]): whole dataframe containing both x and y
        and all one hot encoded input

    Returns:
        [type]: One inter and one 
        text output + onehot encode in one column
    """
    df['Sentiment'] = df['Positive'].astype(
        str) + df['Negative'].astype(str)+df['Neutral'].astype(str)

    sentiment_map = {'100': 1, '001': 0, '010': -1}
    sentiment_word_map = {'100': 'positive',
                          '001': 'neutral', '010': 'negative'}

    df['Sentiment_int'] = df['Sentiment'].map(sentiment_map)
    df['Sentiment_wrd'] = df['Sentiment'].map(sentiment_word_map)
    return df

# Splitting based on criteria


def split_data(X,
               y,
               test_size=0.1,
               shuffle=True):
    """Splitting data based on X and y

    Args:
        X : independent variables
        y : dependent variables
    """
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, y,
        stratify=y,
        random_state=random_number,
        test_size=test_size,
        shuffle=shuffle
    )
    return X_train, X_valid, Y_train, Y_valid


def extract_features(words_train,
                     words_valid,
                     vocabulary_size=5000,
                     Normalize=True):
    """Generate features from words with 
    given Vectorizers


    Args:
        words_train ([list]): Training words
        words_valid ([list]): Validation words
        vocabulary_size (int, optional): 
    """

    tfidv = TfidfVectorizer(
        min_df=3,
        max_features=vocabulary_size,
        preprocessor=lambda x: x, tokenizer=lambda x: x,
        strip_accents='unicode',
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
    )
    feature_train = tfidv.fit_transform(
        words_train,

    ).toarray()
    feature_valid = tfidv.transform(
        words_valid,

    ).toarray()
    vocabulary = tfidv.vocabulary_

    if Normalize:
        feature_train = pr.normalize(
            feature_train)
        feature_valid = pr.normalize(
            feature_valid)
    return feature_train, feature_valid, vocabulary


def _preprocess_xtract(path, name_of_file):

    df = pd.read_excel(path/name_of_file)
    df = preprocess_y_data(df)
    df = df.drop(['ID','Sentiment','Positive', 'Negative', 'Neutral'],
        axis=1)
    X, y = df['Sentence'].values, df['Sentiment_int']
    X = np.array(list(map(review_to_words, X)))
    
    return X, y
