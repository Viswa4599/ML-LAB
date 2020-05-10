from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.decomposition import PCA
import pandas as pd
from itertools import dropwhile
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split as tts
from sklearn.cluster import KMeans
import os
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk import RegexpChunkParser
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import string
from nltk.util import ngrams
import random
from collections import Counter
nltk.download('averaged_perceptron_tagger')
matplotlib.use('agg')

def preprocess(x,n_grams,lower_case=True,punctuation=True,numbers=True,unicode=True,stop_words=True,stemming=True,lemmatizing=True,not_cluster=True):

        stop_words_dictionary = set(map(lambda x: x.lower(), stopwords.words("english")))
        strip_punctuation_translator = str.maketrans("", "", string.punctuation)
        strip_numbers_translator = str.maketrans("", "", string.digits)
        stemmer = PorterStemmer().stem
        lemmatizer = WordNetLemmatizer().lemmatize
        if lower_case:
            description = x.lower()
        if punctuation:
            description = description.translate(strip_punctuation_translator).strip()
        if numbers:
            description = description.translate(strip_numbers_translator).strip()
        if unicode:
            description = description.encode('ascii', 'ignore').decode("utf-8")
        if stop_words:
            word_tokens = word_tokenize(description)
            delete_stop_words = [w for w in word_tokens if not w in stop_words_dictionary]
            description = " ".join(delete_stop_words)
        if stemming:
            word_tokens = word_tokenize(description)
            stemmed = [stemmer(w) for w in word_tokens]
            description = " ".join(stemmed)
        if lemmatizing:
            word_tokens = word_tokenize(description)
            lemmatized = [lemmatizer(w) for w in word_tokens]
            description = " ".join(lemmatized)
        if not_cluster:
            word_tokens = word_tokenize(description)
            tagged = nltk.pos_tag(word_tokens)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #chunked.draw()
            #print(chunked)
            n_gram_done = ngrams(chunked, n_grams)
            #rint(n_grams, ' ARE: '+ [' '.join(grams) for grams in n_gram_done])
            return tagged,chunked,n_gram_done
        
        return description

directory = '20news-19997/20_newsgroups'

#PREPROCESSING
for topic in random.sample(os.listdir(directory),3):
    sub_dir = '20news-19997/20_newsgroups/'+topic
    file_name = random.sample(os.listdir(sub_dir),1)[0]
    with open(sub_dir+'/'+file_name) as f:
        content = f.read().replace('\n', '')
    tagged,chunked,n_gram_done = preprocess(content,n_grams=2)

    print("**********************FILE AFTER TAGGING**************************")
    print(tagged)

    print("**********************FILE AFTER CHUNKING**************************")
    print(chunked)


    print("**********************N GRAMS**************************")
    try:
        print(Counter(n_gram_done))
    except:
        print('Unable to print N GRAM DATA')

print('**&*&*&*&*&*&*&*&*&*&*&*&*&*KMEANS CLUSTERING*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&')
#CLUSTERING


def print_elbow(data,vec_type,fig_count):
    #Elbow Method
    wcss = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        #distortions.append(sum(np.min(cdist(vec_df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vec_df.shape[0])
        wcss.append(kmeanModel.inertia_)

    print(wcss)
    plt.figure(fig_count)
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.title('The '+vec_type+' Elbow Method showing the optimal k')
    plt.savefig('Elbow KMeans '+vec_type)


def visualize(data,vec_type,fig_count):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    preds = kmeans.predict(data)

    colmap = {0:'r',1:'b',2:'g',3:'k',4:'y'}
    label_color = [colmap[l] for l in preds]
    plt.figure(fig_count)
    plt.scatter(data['X'],data['Y'],c = label_color)
    plt.savefig('NEWSGROUPS '+vec_type+' K MEANS PLOT')

#PREPROCESSING BEFORE CLUSTERING
train_data = []
for topic in random.sample(os.listdir(directory),5):
    sub_dir = '20news-19997/20_newsgroups/'+topic
    for filename in random.sample(os.listdir(sub_dir),10):
        with open(sub_dir+'/'+filename) as f:
            content = f.read().replace('\n','')
        train_data.append(content)

train_data_processed = []
for data in train_data:
    train_data_processed.append(preprocess(data,n_grams=2,not_cluster=False))


print('*****************DOC2VEC METHOD******************************')
VEC_TYPE = 'DOC2VEC'

tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(train_data_processed)]
print(tagged_data)
max_epochs = 30
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

vecs = []
for doc in tagged_data:
    vec = model.docvecs[doc[1][0]]
    vecs.append(vec)
    

vec_df = pd.DataFrame(vecs)

#Dimensionality reduction
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(vec_df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['X', 'Y'])

print_elbow(principalDf,VEC_TYPE,1)
visualize(principalDf,VEC_TYPE,2)

print('*****************TFIDF METHOD******************************')
VEC_TYPE = 'TFIDF'

tfidf = TfidfVectorizer()
train_transformed = tfidf.fit_transform(train_data_processed)
train_transformed_dense = train_transformed.todense()
train_PCA = PCA(n_components=2).fit_transform(train_transformed_dense)

principalDf1 = pd.DataFrame(train_PCA,columns = ['X','Y'])


print_elbow(principalDf1,VEC_TYPE,3)
visualize(principalDf1,VEC_TYPE,4)
