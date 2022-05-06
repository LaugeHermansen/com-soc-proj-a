#%%

import re
# from nltk.book import *
import nltk
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import LatentDirichletAllocation


#%%


class MyStemmer(nltk.PorterStemmer):

    """
    This class is an extension of the porter stemmer.
    It adds a function called reverse stem, that can turn a stem
    back into a real word (not necessarily the word it was before, but
    the shortest seen word whose stem is the given - i.e., a real
    word that has a meaning, and is related to the stem)
    """

    def __init__(self, filename = None, *args, **kwargs):
        print("Creating stemmer - trying to read ... ", end = "")
        super().__init__(*args, **kwargs)
        self.filename = filename
        try:
            self.reverse_dict = pd.read_pickle('data/' + filename)
            print("File read sucessfuly - stemmer created")
        except FileNotFoundError:
            print('File not found - created empty stemmer')
            self.reverse_dict = {}
        
    
    def stem(self, *args, **kwargs):
        stem = super().stem(*args, **kwargs)
        if stem not in self.reverse_dict or len(self.reverse_dict[stem]) > len(args[0]) > len(stem):
            self.reverse_dict[stem] = args[0]
        if stem == 'chri' and args[0] == 'chri':
            # raise ValueError("WHAAAAAAT THE FUUUUUUUCK")
            pass
        return stem
    
    def many_stem(self, stems, *args, **kwargs):
        ret = [None]*len(stems)
        for i, stem in enumerate(stems):
            ret[i] = self.stem(stem, *args, **kwargs)
        return ret

    def reverse_stem(self,stem):
        if stem in self.reverse_dict:
            return self.reverse_dict[stem]
        else:
            print(f'Warning: {stem} has not been stemmed before by this stemmer.')
            return stem
    
    def many_reverse_stem(self, stems):
        ret = [None]*len(stems)
        for i,s in enumerate(stems):
            ret[i] = self.reverse_stem(s)
        return ret

    def save(self):
        print("Saving stemmer ... ", end = "")
        pd.to_pickle(self.reverse_dict, 'data/' + self.filename)
        print("Sucessfully saved stemmer")

#%%

def tokenize_stem_remove(text, stemmed_stopwords, stemmer: MyStemmer):
    "tokenize, stem and remove unwanted words - return a set"

    def tokenize_stem_remove_gen(text):
        if text == "[removed]":
            return set()
        try:
            for stemmed_word in map(stemmer.stem, nltk.tokenize.word_tokenize(text.lower())):
                if stemmed_word in stemmed_stopwords:            pass
                elif re.search(r'\W', stemmed_word):             pass
                elif len(stemmed_word) != 0:                     yield stemmed_word
        except ValueError:
            raise ValueError(text)
    return set(tokenize_stem_remove_gen(text))

#%%

#Open data files

def open_data_files(filename, stemmer: MyStemmer, stemmed_stopwords, restart = False):
    def _open_helper(filename):
        filename = filename.replace('.pkl', '')
        if restart: data = pd.read_pickle(f'data/raw/{filename}.pkl')
        else:
            data = pd.read_pickle(f'data/{filename}.pkl')
            print(f"{filename} loaded successfully")
        overwrite = "tokens" not in data.columns
        # promt = f"'tokens' column was found in {filename}\nWanna overwrite? (y/n)"
        # overwrite = input(promt) == "y" if "tokens" in data.columns else True
        if overwrite:
            print(f"Tokenizing texts in {filename} ... ", end = "")
            data['tokens'] = [tokenize_stem_remove(text, stemmed_stopwords, stemmer) for text in tqdm(data['body'])]
            print(f"Done")
        for c in data.columns:
            if 'id' in c and type(data[c][0]) == str and  data[c][0][0] == 't' and data[c][0][2] == "_":
                data[c] = data[c].apply(lambda x: x[3:])
        pd.to_pickle(data, f'data/{filename}.pkl')
        return data

    print("Loading data file ... ")
    data = _open_helper(filename)
    stemmer.save()
    print("Done loading data - stemmer was saved")

    return data



#%%

#Create documents



class Document:
    """
    This is a document class - 
    it contains all the words that has been used relating to the same category
     weihted by how many comments they appear in """
    def __init__(self, old_document = None):
        self.rawcount_tf = defaultdict(int)
        self.n_words = 0
        self.tf = None
        self.tf_idf = None
        self.name = None
        if old_document != None:
            self.read_old(old_document)
    
    def compute_tf(self):
        self.n_words = len(self.rawcount_tf)
        self.tf = {word:count/self.n_words for word, count in self.rawcount_tf.items()}
    
    def compute_tf_idf(self, idf):
        self.compute_tf()
        self.tf_idf = {word: tf*idf[word] for word, tf in self.tf.items()}
    
    def get_wordcloud(self, stemmer):
        unstemmed_tf_idf = {stemmer.reverse_stem(stem): tf_idf for stem, tf_idf in self.tf_idf.items()}
        wordcloud = WordCloud(height=400, width=400, background_color='white').generate_from_frequencies(unstemmed_tf_idf)
        return wordcloud

    def __in__(self, word):
        return word in self.rawcount_tf

    @property
    def words(self):
        return set(self.tf.keys())
    
    def read_old(self, old):
        old_vars = vars(old)
        for name in vars(self):
            try: vars(self)[name] = old_vars[name]
            except: pass
    
    def __repr__(self):
        return f"Document: {self.n_words} words, tf_idf {sorted(self.tf_idf.items(), key = lambda x: x[0])[:4]} ..."
    
    def __len__(self):
        return len(self.tf_idf)


class Corpora:
    """Corpora class
    
    contains documents

    """
    def __init__(self, data, split_by_column, renew = False):

        print(f"Creating corpora with documents = {split_by_column} ... ")

        self.renew = renew
        self.split_by_column = split_by_column
        
        self.load_variable('doc_word_matrix', set_none = True)
        self.load_variable('feature_map', set_none = True)
        self.load_variable('document_map', set_none = True)

        if not self.load_variable('documents'):
            self.documents = defaultdict(Document)
            self.compute_tf_for_each_doc(data)
            for name, document in list(self.documents.items()):
                self.documents[name].name = name

        else:
            for name, document in tqdm(list(self.documents.items())):
                self.documents[name] = Document(document)
                self.documents[name].name = name
        
        self.n_documents = len(self.documents)

        if not self.load_variable('rawcount_df'):
            self.rawcount_df = defaultdict(int)
            for document in tqdm(self.documents.values()):
                for word in document.tf:
                    self.rawcount_df[word] += 1
            print('Done creating rawcount_df')
        
        if not self.load_variable('idf'):
            self.idf = {word: np.log(self.n_documents/df) for word, df in self.rawcount_df.items()}
            for document in self.documents.values():
                document.compute_tf_idf(self.idf)
                print('Done creating idf')
        
        self.n_words = len(self.idf)
        self.save()
        print(f"Done creating/loading corpora {self.split_by_column}")


    def compute_tf_for_each_doc(self, data):
        if self.split_by_column != None:
            for document_name, tokens in tqdm(data[[self.split_by_column, 'tokens']].values):
                if isinstance(document_name, str): document_name = [document_name]
                for doc_name in document_name:
                    for stem in tokens:
                        self.documents[doc_name].rawcount_tf[stem] += 1
                    self.documents[doc_name].compute_tf()
    
    def __repr__(self):
        return f"Corpora with {self.n_words} words and {len(self.documents)} documents"
    
    def path(self):
        if not os.path.exists("data/corpus/"):
            os.mkdir("data/corpus/")
        path = f"data/corpus/Corpora_spit_by_\'{self.split_by_column}\'"
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def save(self, variable_name = None):
        path = self.path()
        if variable_name == None:
            for variable_name, object in vars(self).items():
                pd.to_pickle(object, f"{path}/{variable_name}.pkl")
        else:
            pd.to_pickle(vars(self)[variable_name], f"{path}/{variable_name}.pkl")

    
    def load_variable(self, variable_name, set_none = False):
        path = self.path()
        try:
            assert self.renew == False
            vars(self)[variable_name] = pd.read_pickle(f"{path}/{variable_name}.pkl")
            print(f'Variable {variable_name} loaded sucessfully')
            return True
        except (FileNotFoundError, AssertionError):
            print(f"Couldn't load {variable_name}, ", end = "")
            if set_none:
                vars(self)[variable_name] = None
                print("sat to 'None'")
            else:
                print("creating new one")
            return False

    def __len__(self):
        return len(self.documents)

    def document_sizes(self):
        return np.array(list(map(len, self.documents.values())))
    
    def get_documents(self, n_docs = None, sort = True):
        if sort or n_docs != None:
            return list(reversed(sorted(self.documents.items(), key = lambda x: len(x[1]))))[:n_docs]
        else:
            return list(self.documents.items())
    
    def get_doc_word_matrix(self, doc_min_size = 0, recalculate = False):

        if None in [self.doc_word_matrix,
                    self.feature_map,
                    self.document_map] or recalculate:
            print("Creating document-word matrix ... ")
            
            self.feature_map = {word: i for i, word in enumerate(self.idf)}
            self.document_map = {doc_name: i for i, doc_name in enumerate(self.documents)}

            self.doc_word_matrix = lil_matrix((self.n_documents, len(self.idf)))
            # loop over words and docs
            for doc_name, document in tqdm(self.documents.items()):
                for word in document.words:
                    self.doc_word_matrix[self.document_map[doc_name],self.feature_map[word]] += document.rawcount_tf[word]
            self.save('doc_word_matrix')
            self.save('feature_map')
            self.save('document_map')

        size_mask = [len(document) > doc_min_size for document in self.documents.values()]
        
        assert sum(size_mask) > 0

        if doc_min_size > 0:
            print(f'found {sum(size_mask)} documents with size >= {doc_min_size}')


        r1 = np.array(list(self.feature_map.keys()))
        r2 = np.array(list(self.document_map.keys()))[size_mask]
        r3 = self.doc_word_matrix.tocsr()[size_mask, :]

        return r1, r2, r3
    
    def get_doc_size_distribution(self):
        return np.array(list(map(len, self.get_documents())))



class LDA(LatentDirichletAllocation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y=None, **fit_params):
        print("Fitting doc word matrix ... ", end = "")
        super().fit(X, y, **fit_params)
        print("Done")

    def transform(self, X):
        print("Transforming doc word matrix ... ", end = "")
        ret = super().transform(X)
        print("Done")
        return np.argmax(ret, axis = 1)

