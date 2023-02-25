from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import string


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def single_text_processing(raw_text):
    
    # Split the string into tokens
    text = raw_text.split()
    
    # Lowercasing text
    text = [token.lower() for token in text]
    
    # Removing punctuations from text column
    text = [token.translate(str.maketrans('', '', string.punctuation)) for token in text]
    
    # Removing numbers
    text = [re.sub(r'\d+', '', token) for token in text]
    
    # Removing stopwords
    text = [token for token in text if token and token not in stop_words]
    
    
    return text


def text_preprocessing(dataset, lemmatization=None, stemming=None): 
    if lemmatization and stemming:
        raise Exception('Sorry, select one of these two methods(lemmatization or stemming) to be applied')
    elif not(lemmatization or stemming):
        raise Exception('Sorry, please choose at least one of these two methods(lemmatization or stemming) to be applied')
        
    
    # Convert dataframe column to list for faster computation
    text = dataset['text_column'].tolist()
    
    # Call the single_text_processing method to apply every preprocessing step in one list comprehension for less compuations
    text = [single_text_processing(sentence) for sentence in text]
    
    help_list = []
    
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        for index, sentence in enumerate(text):
            lemmatized_text = []            
            for token in sentence:
                lemmatized_word = lemmatizer.lemmatize(token)
                lemmatized_text.append(lemmatized_word)
            help_list.append(lemmatized_text)
    elif stemming:
        ps = PorterStemmer()
        for index, sentence in enumerate(text):
            stemmed = []
            for token in sentence:
                stemmed_word = ps.stem(token)
                stemmed.append(stemmed_word)
            help_list.append(stemmed)
            

    dataset2 = dataset.copy()
    dataset2['text_column'] = help_list

        
    return dataset2 
    