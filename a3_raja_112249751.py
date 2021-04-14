#!/usr/bin/python3
# Hassan Raja 112249751
# CSE354, Spring 2021
##########################################################
## a3_raja_112249751.py

import sys
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch

sys.stdout = open('a3_raja_112249751_OUTPUT.txt', 'w')

### Part1:

def loadData(filename):
    lines = open(filename, encoding='utf-8').read().splitlines()  # split lines
    liness = []
    for line in lines:
        liness.append(line.split('\t'))  # split each line by tab

    data = []
    for line in liness:
        str1 = line[2]
        str1 = removeHeadTags(str1)
        data.append(tokenize(str1))
    return data


headMatch = re.compile(r'<head>([^<]+)</head>')  # matches contents of head
def removeHeadTags(context:str):
    context = context.lower()
    tokens = context.split()  # get the tokens
    headIndex = -1  # will be set to the index of the target word

    for i in range(len(tokens)):
        m = headMatch.match(tokens[i])
        if m:  # a match: we are at the target token
            tokens[i] = m.groups()[0]
            headIndex = i

    context = ' '.join(tokens)

    return context


def tokenize(string:str):
    tokens = string.split(' ')
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if tokens[i][j] == '/':
                tokens[i] = tokens[i][:j]
                break

    tokens.insert(0, '<s>')
    tokens.append('</s>')

    return tokens


def getWordFrequencies(data):
    freqs = {}
    for sentence in data:
        for word in sentence:
            if word not in freqs:
                freqs[word] = 0
            freqs[word] += 1

    return freqs


def get_top_5000_words(dic):  # trims dictionary to top 5000 word counts
    # sorted_dic = sorted(dic.items(), key = lambda kv: kv[1])
    while len(dic) != 5000:
        min_key = min(dic.keys(), key=lambda k: dic[k])
        del dic[min_key]


def extractUnigramCounts(data, vocab):
    unigramCounts = {}
    for sentence in data:
        for word in sentence:
            if word not in vocab:
                if '<OOV>' not in unigramCounts:
                    unigramCounts['<OOV>'] = 0
                unigramCounts['<OOV>'] += 1
            else:
                if word not in unigramCounts:
                    unigramCounts[word] = 0
                unigramCounts[word] += 1

    return unigramCounts


def extractBigramCounts(data, vocab):
    bigramCounts = {}
    for sentence in data:
        for i in range(len(sentence)-1):
            wiMinus1, wi = sentence[i], sentence[i+1]

            key = "<OOV>" if wiMinus1 not in vocab else wiMinus1

            if key not in bigramCounts:
                bigramCounts[key] = {}
            if wi not in vocab:
                wi = "<OOV>"
            if wi not in bigramCounts[key]:
                bigramCounts[key][wi] = 0
            bigramCounts[key][wi] += 1

    return bigramCounts


def extractTrigramCounts(data, vocab):
    trigramCounts = {}
    for sentence in data:
        for i in range(len(sentence)-2):
            w1, w2, w3 = sentence[i], sentence[i+1], sentence[i+2]  #CHECK IF TUPLE EXISTS
            key = None

            if w1 not in vocab and w2 not in vocab:
                key = ("<OOV>", "<OOV>")
            elif w1 in vocab and w2 not in vocab:
                key = (w1, "<OOV>")
            elif w1 not in vocab and w2 in vocab:
                key = ("<OOV>", w2)
            else:
                key = (w1, w2)

            if key not in trigramCounts:
                trigramCounts[key] = {}
            if w3 not in vocab:
                w3 = "<OOV>"
            if w3 not in trigramCounts[key]:
                trigramCounts[key][w3] = 0
            trigramCounts[key][w3] += 1

    return trigramCounts


def calculateLanguageModelProbabilities(unigram, bigram, trigram, vocab, wordMinus1, wordMinus2=None):
    allWi = list(bigram[wordMinus1].keys())
    # use add-one smoothing on bigram
    smoothedBigramValue = {}
    smoothedBigramValue[wordMinus1] = {}
    for wi in allWi:
        try:
            bigramCnt = bigram[wordMinus1][wi]
        except KeyError:
            bigramCnt = 0
        smoothedBigramValue[wordMinus1][wi] = (bigramCnt+1) / (unigram[wordMinus1] + len(vocab))

    if not wordMinus2:
        return smoothedBigramValue

    key = (wordMinus2, wordMinus1)
    smoothedTrigramValue = {}
    smoothedTrigramValue[key] = {}
    for wi in allWi:
        try:
            trigramCnt = trigram[key][wi]
        except KeyError:
            trigramCnt = 0
        try:
            bigramCnt1 = bigram[wordMinus1][wi]
        except KeyError:
            bigramCnt1 = 0
        smoothedTrigramValue[key][wi] = (trigramCnt+1) / (bigramCnt1 + len(vocab))
        try:
            smoothedTri = smoothedTrigramValue[key][wi]
        except:
            smoothedTri = 0
        try:
            smoothedBi = smoothedBigramValue[wordMinus1][wi]
        except KeyError:
            smoothedBi = 0
        smoothedTrigramValue[key][wi] = (smoothedBi + smoothedTri) / 2

    return smoothedTrigramValue


def getProbs(dic):
    vals = []
    for key in dic:
        vals.append(dic[key])
    return vals


def normalizeList(probs):
    sum1 = sum(probs)
    for i in range(len(probs)):
        probs[i] /= sum1



def generateLanguage(words, unigram, bigram, trigram, vocab):
    sentence = []
    for w in words:
        sentence.append(w)
    numWords = len(sentence)
    if numWords == 1:
        w1 = sentence[0]
        bigram1 = calculateLanguageModelProbabilities(unigram, bigram, trigram, vocab, w1)
        wordsAfter = list(bigram1[w1].keys())
        wordAfterProb = getProbs(bigram1[w1])
        normalizeList(wordAfterProb)
        wordChoice = np.random.choice(wordsAfter, 1, p=wordAfterProb)
        sentence.append(wordChoice[0])
        numWords += 1

    if sentence[-1] == "</s>":
        return ' '.join(sentence)

    while numWords < 32:
        wMinus1 = sentence[-1]
        wMinus2 = sentence[-2]
        key = (wMinus2, wMinus1)
        trigram1 = calculateLanguageModelProbabilities(unigram, bigram, trigram, vocab, wMinus1, wordMinus2=wMinus2)
        wordsAfter = list(trigram1[key].keys())
        wordAfterProb = getProbs(trigram1[key])
        normalizeList(wordAfterProb)
        wordChoice = np.random.choice(wordsAfter, 1, p=wordAfterProb)
        sentence.append(wordChoice[0])
        if sentence[-1] == "</s>":
            break
        numWords += 1

    return ' '.join(sentence)




###################################################################################
## MAIN


if __name__ == "__main__":
    ## Part 2
    ### 2.1
    data = loadData("onesec_train.tsv")
    word_freqs = getWordFrequencies(data)
    # freqsByCount = dict(sorted(word_freqs.items(), key = lambda kv:kv[1], reverse = True))
    freqs = dict(sorted(word_freqs.items(), key = lambda kv:kv[0]))
    get_top_5000_words(freqs)
    vocab = list(freqs.keys())

    ### 2.2
    unigramCounts = extractUnigramCounts(data, vocab)
    bigramCounts = extractBigramCounts(data, vocab)
    trigramCounts = extractTrigramCounts(data, vocab)


    print("Unigrams:")
    try:
        print("'language': " + str(unigramCounts['language']))
    except KeyError:
        print("'language': 0")
    try:
        print("'the': " + str(unigramCounts['the']))
    except KeyError:
        print("'the': 0")
    try:
        print("'formal': " + str(unigramCounts['formal']))
    except KeyError:
        print("'formal': 0")
    print("\nBigrams:")
    try:
        print("('the','language'): " + str(bigramCounts['the']['language']))
    except KeyError:
        print("('the','language'): 0")
    try:
        print("('<OOV>','language'): " + str(bigramCounts['<OOV>']['language']))
    except KeyError:
        print("('<OOV>','language'): 0")
    try:
        print("('to','process'): " + str(bigramCounts['to']['process']))
    except KeyError:
        print("('to','process'): 0")
    print("\nTrigrams:")
    try:
        print("('specific','formal','languages'): " + str(trigramCounts[('specific', 'formal')]['languages']))
    except KeyError:
        print("('specific','formal','languages'): 0")
    try:
        print("('to','process','<OOV>'): " + str(trigramCounts[('to', 'process')]['<OOV>']))
    except KeyError:
        print("('to','process','<OOV>'): 0")
    try:
        print("('specific','formal','event'): " + str(trigramCounts[('specific', 'formal')]['event']))
    except KeyError:
        print("('specific','formal','event'): 0")


    ### 2.3
    print("\n2.3\nBigrams:")
    smoothedBigram1 = calculateLanguageModelProbabilities(unigramCounts, bigramCounts, trigramCounts, vocab, "the")
    smoothedBigram2 = calculateLanguageModelProbabilities(unigramCounts, bigramCounts, trigramCounts, vocab, "<OOV>")
    smoothedBigram3 = calculateLanguageModelProbabilities(unigramCounts, bigramCounts, trigramCounts, vocab, "to")

    try:
        print("P('the','language'): " + str(smoothedBigram1['the']['language']))
    except KeyError:
        print("P('the','language'): Not valid Wi")
    try:
        print("P('<OOV>','language'): " + str(smoothedBigram2['<OOV>']['language']))
    except KeyError:
        print("P('<OOV>','language'): Not valid Wi")
    try:
        print("P('to','process'): " + str(smoothedBigram3['to']['process']))
    except KeyError:
        print("('to','process'): Invalid Wi ")

    print("\nTrigrams:")
    smoothedTrigram1 = calculateLanguageModelProbabilities(unigramCounts, bigramCounts, trigramCounts, vocab,
                                                           wordMinus1="formal", wordMinus2="specific")
    smoothedTrigram2 = calculateLanguageModelProbabilities(unigramCounts, bigramCounts, trigramCounts, vocab,
                                                           wordMinus1="process", wordMinus2="to")

    try:
        print("P('specific','formal','languages'): " + str(smoothedTrigram1[('specific', 'formal')]['languages']))
    except KeyError:
        print("P('specific','formal','languages'): Not valid Wi")
    try:
        print("P('to','process','<OOV>'): " + str(smoothedTrigram2[('to', 'process')]['<OOV>']))
    except KeyError:
        print("P('to','process','<OOV>'): Not valid Wi")
    try:
        print("P('specific','formal','event'): " + str(smoothedTrigram1[('specific', 'formal')]['event']))
    except KeyError:
        print("P('specific','formal','event'): Not valid Wi")

    ### 2.4
    l1 = ['<s>']
    l2 = ['<s>', 'language', 'is']
    l3 = ['<s>', 'machines']
    l4 = ["<s>", 'they', 'want', 'to', 'process']
    print("\nPROMPT: <s>")
    for i in range(3):
        print(generateLanguage(l1, unigramCounts, bigramCounts, trigramCounts, vocab))

    print("\nPROMPT: <s> language is")
    for i in range(3):
        print(generateLanguage(l2, unigramCounts, bigramCounts, trigramCounts, vocab))

    print("\nPROMPT: <s> machines")
    for i in range(3):
        print(generateLanguage(l3, unigramCounts, bigramCounts, trigramCounts, vocab))

    print("\nPROMPT: <s> they want to process")
    for i in range(3):
        print(generateLanguage(l4, unigramCounts, bigramCounts, trigramCounts, vocab))
