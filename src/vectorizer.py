import vocabulary
import statistics
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math

class Vectorizer:
    
    def __init__(self, dataset=None):
        
        if dataset is None:
            raise Warning('Sorry I am useless without a dataset with one <<text_column>> column and one <<class_column>> column')
        
        self.vocabulary = vocabulary.Vocabulary(dataset)
        self.tokenized_text_list = dataset['text_column'].tolist()
        self.text_list_with_indexes = []
        self.mean = 0
        self.median = 0
        self.padded_sequences = []
        self.padded_text = []
        
        
    
    @staticmethod
    def convert_words_to_index(word_to_idx_dict,single_row_text):
         return [word_to_idx_dict[word] for word in single_row_text]

    
    # text_list_with_indexes is the initial text column having converted words to indexes
    def convert_text_to_sequences(self):
        for row in self.tokenized_text_list:
            self.text_list_with_indexes.append(self.convert_words_to_index(self.vocabulary.word_to_idx,row))
            

    # Function to calculate the mean number of words per dataset in order to determine the len of the padding sequence
    def calculate_mean_and_median_number_of_words_per_text_per_dataset(self):
        sum_of_words_list = []
        for row in self.text_list_with_indexes:
            sum_of_words_list.append(len(row))
        self.mean = int(sum(sum_of_words_list)/len(self.text_list_with_indexes))
        self.median = int(statistics.median(sum_of_words_list))
            
    def __padding_single_text(self, single_row_text, sequence_length=20):
        if len(single_row_text) < sequence_length:
            single_row_text = single_row_text + [self.vocabulary.word_to_idx['PAD'] for i in range(sequence_length-len(single_row_text)-2)]
            single_row_text = [self.vocabulary.word_to_idx['SOS']] + single_row_text + [self.vocabulary.word_to_idx['EOS']]  
        else:
            single_row_text = [self.vocabulary.word_to_idx['SOS']] + single_row_text[:sequence_length-2] + [self.vocabulary.word_to_idx['EOS']]
        return single_row_text
    
    def text_to_padded_sequences(self):
        for row in self.text_list_with_indexes:
            self.padded_sequences.append(self.__padding_single_text(row))
    
######################################################################################################################################################

    def _padding_single_text_to_real_words(self, single_row_text, sequence_length=20):
        if len(single_row_text) < sequence_length:
            single_row_text = single_row_text + [self.vocabulary.idx_to_word[0] for i in range(sequence_length-len(single_row_text)-2)]
            single_row_text = [self.vocabulary.idx_to_word[1]] + single_row_text + [self.vocabulary.idx_to_word[2]]  
        else:
            single_row_text = [self.vocabulary.idx_to_word[1]] + single_row_text[:sequence_length-2] + [self.vocabulary.idx_to_word[2]]
        return single_row_text
    
    def padding_text(self):
        for row in self.tokenized_text_list:
            self.padded_text.append(self._padding_single_text_to_real_words(row))
            
######################################################################################################################################################  
     
            
            
class BowVectorizer(Vectorizer):
    
    def __init__(self, dataset=None):
        super().__init__(dataset)
        
        self.dataset = dataset
        
        
    def vectorize_text_to_bow_and_create_dataframe(self,return_array=False):         

        unique_words = set(self.vocabulary.unique_words)        
        bow_array = np.zeros((len(self.tokenized_text_list), len(unique_words)))

        for row,text in enumerate(self.tokenized_text_list):
            bow_array[row] = [1 if word in text else 0 for word in unique_words]
            
        bow_df = pd.DataFrame(
            data=bow_array,
            index=[doc for doc in self.tokenized_text_list],
            columns=[column for column in unique_words]            
            )
        
        bow_df['class_column'] = self.dataset['class_column'].tolist()

        if return_array == True:
            return bow_df, bow_array
        else:
            return bow_df
        
    
        
        

class TfIdfVectorizer(Vectorizer):
    
    def __init__(self, dataset=None):
        super().__init__(dataset)
        
        # Convert text column to a sequence of preprocessed words in order to feed it in sklearn's fit_transform method 
        self.dataset = dataset
        self.tfidf_text = [' '.join(text) for text in self.tokenized_text_list]
        
        
    def create_dataframe(self):
        
        
        vectorizer = TfidfVectorizer() 
        # TD-IDF Matrix
        X = vectorizer.fit_transform(self.tfidf_text)
        # extracting feature names
        tfidf_tokens = vectorizer.get_feature_names_out()
        
        tf_idf_dataset = pd.DataFrame(
            data=X.toarray(), 
            index=[doc for doc in self.tfidf_text], 
            columns=tfidf_tokens
        )
        
        tf_idf_dataset["class_column"] = self.dataset['class_column'].tolist()
        
        return tf_idf_dataset
        
    
        
        
        
        
            
        
            
            
            
        
        
        

