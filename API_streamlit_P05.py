import streamlit as st

import pandas as pad 
import numpy as np

## nettoyage
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from bs4 import BeautifulSoup
import spacy
from spacy.lemmatizer import Lemmatizer
import gensim
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.utils import simple_preprocess

from joblib import dump, load

nltk.download('stopwords')

import en_core_web_sm



def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    streamlit.title('Prédiction de tags pour question stack overflow')

    post = streamlit.text_input('Question stack overflow', "aggressive javascript cach've run problem make changes javascript files referenced html file browser")

    data_words = post

    # Build the bigram model
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # NLTK Stop words
    stop_words = stopwords.words('english')
    stop_words.extend(["?", "How", "What", "way", "Why", "I", "(", ")", ",,", "Is", ":", "'s", "``", "-", "best", "Best",'<', '>', '/p', 'from', 'subject', 're', 'edu', "pre","strong","want","use","add","work","get","attach","try","miss","complain","readable","cleanly","unknown","detach","obviously","miserably","absolutely","edit","probably","better","question","come","know","follow","holiday","new","successful","lastname","user","suppose","clear","seem","involve","recieve","sure","make","really","recommendation","need","specifically","detail","flat","do","slurp","thank","look","recommend","take","also","continue","may","would","option","like","put","true"])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def clean_text(texts):
        texts = re.sub(r"\'", "'", texts)
        texts = re.sub(r"\n", " ", texts)
        texts = re.sub(r"\xa0", " ", texts)
        texts = re.sub('\s+', ' ', texts)
        texts = texts.strip(' ')
        return texts

    def remove_stopwords(texts):
        return [word for word in simple_preprocess(str(texts)) if word not in stop_words]

    def make_bigrams(texts):
        return bigram_mod[texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        doc = nlp(" ".join(texts)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # First clean
    data_words = clean_text(data_words)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    #nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    nlp = en_core_web_sm.load()

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    df_posts = pad.DataFrame()
    df_posts["Title_and_Body"] = data_lemmatized
    df_posts['Title_and_Body'] = df_posts['Title_and_Body'].apply(lambda x: ', '.join(x))
    df_posts['Title_and_Body'] = df_posts['Title_and_Body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    def clean(df):
        df = df.replace('< /p >','')
        df = df.replace('< /code >','')
        df = df.replace('< /pre >','')
        df = df.replace('< pre >','')
        df = df.replace('< code >','')
        df = df.replace('<','')
        df = df.replace('>','')
        df = df.replace('!','')
        df = df.replace("'ve",'')
        df = df.replace('.','')
        df = df.replace('//','')
        df = df.replace('/a','')
        df = df.replace('/blockquote','')
        df = df.replace('/li','')
        df = df.replace('/ul','')
        df = df.replace('0','')
        df = df.replace(';','')
        df = df.replace('=','')
        df = df.replace(']','')
        df = df.replace('[','')
        return df

    df_posts['Document'] = df_posts['Title_and_Body'].apply(clean)

    data_words = df_posts['Document'].values.tolist()
    post = data_words[0].split(', ')


    ## Préparation des données
    dictionary = load('dictionary.joblib')
    datawords_bow = dictionary.doc2bow(post, allow_update=False) 
    datawords_Sparse = corpus2csc([datawords_bow], num_terms=len(dictionary)).transpose()
    post_sup = datawords_Sparse


    def unsupervised_model_prediction_tags(sentence):
        new_corpus = [id2word.doc2bow(text) for text in [sentence]]
        proba = []
        for elm in lda_model.get_document_topics(new_corpus)[0]:
            proba.append(elm[1])

        top_topic_index = proba.index(max(proba))
        top_topic = lda_model.get_document_topics(new_corpus)[0][top_topic_index][0]
        return df_results_unsupervised.loc[top_topic].values



    lda_model = load('LDA_model.joblib')
    id2word = load('id2word.joblib')
    df_results_unsupervised = load('df_results_unsupervised.joblib')


    model_svm = load('SVM_model.joblib')
    binarizer = load('binarizer.joblib')

    predict_btn = streamlit.button('Prédire')
    if predict_btn:
        #data = [[post]]
        #data = post.toJSON()
        pred = None


        ## Unsupervised prediction ##
        streamlit.write('Pour le modèle non-supervisé (LDA) : \n')
        streamlit.write('Les tags prédis sont : ' + str(unsupervised_model_prediction_tags(post)))

        ## Supervised prediction #
        pred = model_svm.predict(post_sup)
        #pred = request_prediction(MLFLOW_URI, post)[0] * 100000
        tags_supervised = binarizer.inverse_transform(pred)
        #st.write('Les tags prédis sont : {:.2f}'.format(tags_supervised))
        streamlit.write('\n\nPour le modèle supervisé (SVM) : \n')
        streamlit.write("Les tags prédis sont : ")
        streamlit.write(str(tags_supervised))


#####  get the error :

if __name__ == '__main__':
    main()