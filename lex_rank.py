import sys
import numpy as np
import fasttext as ft
from scipy.spatial import distance
from pythonrouge.pythonrouge import Pythonrouge
from sumeval.metrics.rouge import RougeCalculator

import spacy

system_path="/Users/ryousuke/desktop/nlp/summarization/scientific_paper/system_sum/"
reference_path="/Users/ryousuke/desktop/nlp/summarization/scientific_paper/golden_sum/"

"""
Input:

CosineMatrix: 隣接行列
N: 入力文数
err_tol: PowerMethodにより収束したと判定するための誤差許容値
Output:

p: 固有ベクトル (LexRankスコア)
"""
def PowerMethod(CosineMatrix, N, err_tol):

    p_old = np.array([1.0/N]*N)
    err = 1

    #一様に並んでいるものをCosinMatrixをかけることで重みをかけることに対応する
    while err > err_tol:
        err = 1
        p = np.dot(CosineMatrix.T, p_old)
        err = np.linalg.norm(p - p_old)
        p_old = p
    return p

def word2id(sentences, word_id):

    for sent in sentences:
        for w in sent.strip(".").split():
            if w not in word_id:
                word_id[w] = len(word_id)
    return word_id

def compute_tf(sentences, word_id):

    tf = np.zeros([len(sentences), len(word_id)])

    for i,sent in enumerate(sentences):
        for w in sent.strip(".").split():
            tf[i][word_id[w]] += 1
    return tf

def compute_df(sentences, word_id):

    df = np.zeros(len(word_id))

    for sent in sentences:
        exist = {}
        for w in sent.strip(".").split():
            if w not in exist:
                df[word_id[w]] += 1
                exist[w] = 1
    return df

def compute_idf(sentences, word_id):

    idf = np.zeros(len(word_id))
    df = compute_df(sentences, word_id)

    for i in range(len(df)):
        idf[i] = np.log(len(sentences)/df[i]) + 1
    return idf

def compute_tfidf(sentences):

    word_id = {}

    word_id = word2id(sentences, word_id)
    tf = compute_tf(sentences, word_id)
    idf = compute_idf(sentences, word_id)

    tf_idf = np.zeros([len(sentences), len(word_id)])

    for i in range(len(sentences)):
        tf_idf[i] = tf[i] * idf
    return tf_idf

def compute_cosine(v1, v2):
    return 1 - distance.cosine(v1, v2)

"""
Input:

sentneces: 入力文のリスト
N: 入力文数
threshold: 隣接行列(類似度グラフ)を作成する際の類似度の閾値
vectorizer: 文のベクトル化の手法(tf-idf/word2vec)

Output:

L: LexRankのスコア(各文章の重要度)
"""
def lexrank(sentences, N, threshold, vectorizer):

    CosineMatrix = np.zeros([N, N])
    degree = np.zeros(N)
    L = np.zeros(N)

    if vectorizer == "tf-idf":
        vector = compute_tfidf(sentences)
    elif vectorizer == "word2vec":
        vector = compute_word2vec(sentences)

    # Computing Adjacency Matrix
    for i in range(N):
        for j in range(N):
            CosineMatrix[i,j] = compute_cosine(vector[i], vector[j])
            if CosineMatrix[i,j] > threshold:
                CosineMatrix[i,j] = 1
                degree[i] += 1 #そのセンテンスの重要度をあげる
            else:
                CosineMatrix[i,j] = 0
    # Computing LexRank Score
    for i in range(N):
        for j in range(N):
            CosineMatrix[i,j] = CosineMatrix[i,j] / degree[i]
    L = PowerMethod(CosineMatrix, N, err_tol=10e-6)

    return L

def summarize(sentences):
    sentences = [sent for sent in sentences if sent !="" and len(sent)>5]
    l = lexrank(np.array(sentences),len(sentences),0.3,"tf-idf")
    idx = np.argsort(l)
    try:
        top5 = sorted(idx)[-12]
    except:
        top5 = sorted(idx)[0]
    idx = idx[idx>=top5]
    final_sents = [sentences[i] for i in idx]
    return final_sents


spa = spacy.load("en")
rouge = RougeCalculator(stopwords=True,lang="en",tokenizer=spa)

def evaluate(system_summary,reference_summary):
    rouge_1 = rouge.rouge_n(
                summary=system_summary,
                references=reference_summary,
                n=1,alpha=0)
    rouge_2 = rouge.rouge_n(
                summary=system_summary,
                references=reference_summary,
                n=2,alpha=0)

    rouge_l = rouge.rouge_l(
                summary=system_summary,
                references=reference_summary,
                alpha=0)
    return rouge_1,rouge_2,rouge_l, {"rouge_1:":rouge_1,"rouge_2":rouge_2,"rouge_l":rouge_l}


##前処理
# from ast import literal_eval
import json
data_path="/Users/ryousuke/desktop/nlp/summarization/scientific_paper/arxiv-release/"
f = open(data_path+"val.txt").readlines()
data = list(map(lambda x:json.loads(x),f))
n_data = len(data)
print('n_data',n_data)

t_rouge_1=0
t_rouge_2=0
t_rouge_l=0

for i in range(n_data):
    article = data[i]["article_text"]
    reference_summary  = "".join(data[i]["abstract_text"])
    system_summary = "".join(summarize(article))
    print()
    print("****system_summary****")
    print("article word_size:",sum(list(map(lambda x:len(x.split()),article))))
    print("system_summary word_size:",sum(list(map(lambda x:len(x.split()),system_summary))))
    print("article")
    print(article)
    exit()

    # for i,s in enumerate(system_summary.split(".")):
    #     print(str(i)+":",s)
    rouge_1,rouge_2,rouge_l,score = evaluate(system_summary,reference_summary)
    t_rouge_1+=rouge_1
    t_rouge_2+=rouge_2
    t_rouge_l+=rouge_l
    print()
    print("****reference_summary****")
    # print("".join(reference_summary))
    print("----result-----",score)

    if i>0 and i % 500 == 0:
        print("Average:","rouge1",t_rouge_1/i,"rouge2",t_rouge_2/i,"rouge_l",t_rouge_l/i)
print("Average:","rouge1",t_rouge_1/i,"rouge2",t_rouge_2/i,"rouge_l",t_rouge_l/i)





