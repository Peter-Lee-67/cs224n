#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    (word2vec模型的朴素Softmax损失及梯度函数)

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    (使用中心词和外部词的词嵌入实现朴素softmax损失和梯度计算。这将成为我们搭建word2vec模型的模块。
    对于那些不熟悉numpy概念的人，请注意一个形状为(x, )的numpy数组是一个一维数组，你可以将其当作一个
    长度为x的向量高效地使用)

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    (参数：
    centerWordVec -- numpy数组，是中心词的嵌入，形状为(词向量长度，)，在pdf作业中记作v_c
    outsideWordIdx -- 整数，外部词的索引，在pdf作业中是u_o的o
    outsideVectors -- 所有词汇表中单词的外部词向量，形状为(词汇表中单词数，词向量长度)，是pdf作业中U的转置
    dataset -- 在负采样中使用，此处没有用到)

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)

    (返回：
    loss -- 朴素softmax损失
    gradCenterVec -- 损失函数关于中心词向量的梯度，形状为(词向量长度，)，是pdf作业中的dJ/dv_c
    gradOutsideVecs -- 损失函数关于所有外部词向量的梯度，形状为(词汇表中单词数，词向量长度)，是pdf作业中的(dJ / dU))
    """

    ### YOUR CODE HERE (~6-8 Lines)
    outsideWordVec = outsideVectors[outsideWordIdx, :]  # (word vector length, )
    # softmax
    temp = softmax(outsideVectors @ centerWordVec)  # (num words in vocab, )
    loss = -np.log(temp[outsideWordIdx])
    gradCenterVec = temp @ outsideVectors - outsideWordVec # (word vector length, )
    gradOutsideVecs = np.outer(temp, centerWordVec)  # (num words in vocab, word vector length)
    gradOutsideVecs[outsideWordIdx, :] -= centerWordVec
    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.
    ### 请使用提供的softmax函数，已经在文件开头从外部导入
    ### 这个数值稳定的实现帮助你避免整数溢出的问题

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models
    (word2vec模型的负采样损失函数)

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.
    (计算中心词向量和外部词向量的负采样损失和相应的梯度，以此作为word2vec的模块，K是负样本的采样数量)

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.
    (注意：同一个词可能会被负采样多次，例如如果一个外部词被采样两次，你应当将这个词的梯度翻倍，如果被采样三次，
    梯度乘3，采样4次同理)

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    (参数/返回 说明：与naiveSoftmaxLossAndGradient相同)
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    # 负采样的单词已经为你准备好了，如果你希望和自动求导的结果匹配并拿到分数，请不要修改此处代码
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices   # list, length = K + 1

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.
    # 请在此处使用你实现过的sigmoid函数
    # loss: float, negative sampling loss
    # gradCenterVec: ndarray, (word vector length, )
    # gradOutsideVecs: ndarray, (num words in vocab, word vector length)
    # 注意：负采样的单词可能相同
    transU = outsideVectors[indices, :]
    transU[1:] = -transU[1:]
    onesVec = np.ones(K + 1)
    temp = transU @ centerWordVec
    loss = -np.sum(np.log(sigmoid(temp)))
    gradCenterVec = -(transU.T @ (onesVec - sigmoid(temp)))
    uniqueElem, idx, counts = np.unique(np.array(indices), return_counts=True, return_index=True)   # 统计频次
    gradU = np.outer(onesVec - sigmoid(temp), centerWordVec)
    gradU[0] = -gradU[0]
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[uniqueElem] = gradU[idx] * counts.reshape(-1, 1)
    ### END YOUR CODE
    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec
    (word2vec中的Skip-gram模型)

    Implement the skip-gram model in this function.
    (在此函数中实现skip-gram模型)

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.
    (参数：
    currentCenterWord -- 当前中心词的字符串
    windowSize -- 整数，上下文窗口大小
    outsideWords -- 外部单词，长度不超过2个窗口大小的字符串列表
    word2Ind -- 将单词映射到单词向量列表索引的字典
    centerWordVectors -- 中心词向量(按行排列)，形状为(词汇表单词数，词向量长度)，涉及词汇表中所有的单词，是pdf作业中的V
    outsideVectors -- 外部词向量，形状为(词汇表单词数，词向量长度)，涉及词汇表中所有的单词，是pdf作业中的U
    word2vecLossAndGradient -- 给定外部词向量后对于一个预测向量的损失和梯度函数，可能是naive-softmax或者neg-sample
    )

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    (返回：
    loss -- skip-gram模型的损失函数值，是pdf作业中的J
    gradCenterVecs -- 关于中心词向量的梯度，形状为(词汇表单词数，词向量长度)，是pdf作业中的 dJ / dv_c
    gradOutsideVecs -- 关于所有外部词向量的梯度，形状为(词汇表单词数，词向量长度)，是pdf作业中的 dJ / dU
    )
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    centerWordVec = centerWordVectors[word2Ind[currentCenterWord]]
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        subloss, subGradCenterVec, subGradOutsideVecs = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += subloss
        gradCenterVecs[word2Ind[currentCenterWord]] += subGradCenterVec  # 形状不一致
        gradOutsideVectors += subGradOutsideVecs
    """
    naiveSoftmaxLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset
    )
    negSamplingLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset,
        K=10
    )
    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """
    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad

def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    assert np.allclose(sigmoid(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens

def test_naiveSoftmaxLossAndGradient():
    """ Test naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'naiveSoftmaxLossAndGradient':
        test_naiveSoftmaxLossAndGradient()
    elif args.function == 'negSamplingLossAndGradient':
        test_negSamplingLossAndGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()
