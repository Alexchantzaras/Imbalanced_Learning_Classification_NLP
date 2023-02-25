class Vocabulary:
    
    def __init__(self, dataset=None):
        
        """
        Args:
             tokenized_text_list (list): the pre-­existing list of raw text passed as list in the class
        """
        tokenized_text_list = dataset['text_column']
        
        if tokenized_text_list is None:
            raise Warning('It is necessary to pass the initial text as tokenized list in order to build the Vocabulary')
        else: 
            self.word_to_idx = {'PAD':0,'SOS':1,'EOS':2}
            self.word_to_count = {}
            self.idx_to_word = {0:'PAD', 1:'SOS', 2:'EOS'}
            self.word_number = 3 # The number of initial words in the word_to_idx dictionary
            self._list_of_words = [item for sublist in tokenized_text_list for item in sublist]
            self.unique_words = list(self.idx_to_word.values()) + list(set(self._list_of_words))
            for sentence in tokenized_text_list:
                for token in sentence:
                    if token not in self.word_to_idx:
                        self.word_to_idx[token] = self.word_number
                        self.word_to_count[token] = 1
                        self.idx_to_word[self.word_number] = token
                        self.word_number += 1
                    else:
                        self.word_to_count[token] += 1  
