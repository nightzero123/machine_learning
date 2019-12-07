import scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import re #regular expression for e-mail processing

# 这是一个可用的英文分词算法(Porter stemmer)
from stemming.porter2 import stem

# 这个英文算法似乎更符合作业里面所用的代码，与上面效果差不多
import nltk, nltk.stem.porter

with open('data/emailSample1.txt', 'r') as f:
    email = f.read()
    print(email)

def processEmail(email):
    """做除了Word Stemming和Removal of non-words的所有处理"""
    email = email.lower()
    email = re.sub('<[^<>]>', ' ', email)  # 匹配<开头，然后所有不是< ,> 的内容，知道>结尾，相当于匹配<...>
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email )  # 匹配//后面不是空白字符的内容，遇到空白字符则停止
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[\$]+', 'dollar', email)
    email = re.sub('[\d]+', 'number', email)
    return email


def email2TokenList(email):
    """预处理数据，返回一个干净的单词列表"""

    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()

    email = processEmail(email)

    # 将邮件分割为单个单词，re.split() 可以设置多种分隔符
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # 遍历每个分割出来的内容
    tokenlist = []
    for token in tokens:
        # 删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token);
        # Use the Porter stemmer to 提取词根
        stemmed = stemmer.stem(token)
        # 去除空字符串‘’，里面不含任何字符
        if not len(token): continue
        tokenlist.append(stemmed)

    return tokenlist

def email2VocabIndices(email, vocab):
    """提取存在单词的索引"""
    token = email2TokenList(email)
    index = [i for i in range(len(vocab)) if vocab[i] in token ]
    return index

def email2FeatureVector(email):
    """
    将email转化为词向量，n是vocab的长度。存在单词的相应位置的值置为1，其余为0
    """
    df = pd.read_table('data/vocab.txt',names=['words'])
    vocab = df.as_matrix()  # return array
    vector = np.zeros(len(vocab))  # init vector
    vocab_indices = email2VocabIndices(email, vocab)  # 返回含有单词的索引
    # 将有单词的索引置为1
    for i in vocab_indices:
        vector[i] = 1
    return vector

vector = email2FeatureVector(email)
print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))



# Training set
mat1 = loadmat('data/spamTrain.mat')
X, y = mat1['X'], mat1['y']

# Test set
mat2 = scipy.io.loadmat('data/spamTest.mat')
Xtest, ytest = mat2['Xtest'], mat2['ytest']

clf = svm.SVC(C=0.1, kernel='linear')
clf.fit(X, y)

predTrain = clf.score(X, y)
predTest = clf.score(Xtest, ytest)
predTrain, predTest



