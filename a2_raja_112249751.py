#!/usr/bin/python3
# Hassan Raja 112249751
# CSE354, Spring 2021
##########################################################
## a2_raja_112249751.py
## Ambiguities hiding in plain sight! Have you noticed that the words "language", "process", and "machine", some of the
#  most frequent words mentioned in this course, are quite ambiguous themselves? You will use a subset of a modern word
#  sense disambiguation corpus, called "onesec" for your training and test data.

import sys
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch

sys.stdout = open('a2_raja_112249751_OUTPUT.txt', 'w')

#########################################################
## Part 1. Read and tokenize data.


def loadData(filename):

    lines = open(filename, encoding='utf-8').read().splitlines() # split lines
    liness = []
    for line in lines:
        liness.append(line.split('\t')) # split each line by tab

    data = []  # will contain tuples
    for line in liness:
        element = [line[0], line[1]]  # [ id, sense, context ]
        str1 = ""
        i = 2
        while(i < len(line)):
            str1 += line[i]
            i+=1
        element.append(str1)
        data.append(element)

    target_words = []
    senses = {}
    count = 0
    for x in data:
        temp = ''
        if x[1] not in senses:
            senses[x[1]] = count
            count += 1
            for i in range(len(x[1])):
                if x[1][i] == '%':
                    temp = x[1][:i]
                    if temp not in target_words: target_words.append(temp)

    return data, senses, target_words


headMatch = re.compile(r'<head>([^<]+)</head>')  # matches contents of head
def removeLemmaPOS(data):
    for x in data: # for each context line
        sentence = ''
        words = x[2].split(' ')
        for w in words:
            if '<head>' in w:
                sentence += w
                sentence += ' '
                continue
            for i in range(len(w)):
                if w[i] == '/':
                    sentence += w[:i]
                    sentence += ' '
                    break

        x[2] = sentence


def return_sense_of_target_word(context):
    headMatch = re.compile(r'<head>([^<]+)</head>')  # matches contents of head
    tokens = context.split()  # get the tokens
    headIndex = -1  # will be set to the index of the target word
    for i in range(len(tokens)):
        m = headMatch.match(tokens[i])
        if m:  # a match: we are at the target token
            tokens[i] = m.groups()[0]
            headIndex = i
            sense = tokens[i]
            for j in range(len(sense)):
                if sense[j] == '/':
                    sense = sense[:j]
                    return sense

    return ''


# wordRE = re.compile(r'[\?!\.,\";]|(?:[\w-]+(?:\'[a-z]{1,3}\b)?)', re.UNICODE) # COMPLETE THIS


def getWordFrequencies(context, dic):  # gets the frequency of each word
    words = context.split(' ')
    for w in words:
        w = w.lower()  # lowercase all words
    for w in words:
        if '<head>' in w:
            temp = return_sense_of_target_word(context)
            if temp not in dic:
                dic[temp] = 1
            else:
                dic[temp] += 1
        elif w in dic:
            dic[w] += 1
        else:
            dic[w] = 1


def get_top_2000_words(dic):  # trims dictionary to top 2000 word counts
    # need to remove 10872
    # sorted_dic = sorted(dic.items(), key = lambda kv: kv[1])
    while len(dic) != 2000:
        min_key = min(dic.keys(), key=lambda k: dic[k])
        del dic[min_key]


#########################################################
## Part 3. Logistic Regression Classification

def getTargetExamples(data):
    target_examples = {}
    for x in data:
        key = x[1]
        for j in range(len(x[1])):
            if x[1][j] == '%':
                key = x[1][:j]
        if key not in target_examples:
            target_examples[key] = []
        target_examples[key].append(x)
        continue  # move onto next example

    return target_examples


def getUniqueIds(target_examples):
    ids = []
    for key in target_examples:
        dic = {}
        count = 0
        for row in target_examples[key]:
            if row[1] not in dic:
                dic[row[1]] = count
                count += 1
        ids.append(dic)
    return ids

def removeHeadTagAndGetOneHot(context, vocab):
    prevOneHot, nextOneHot = [0] * len(vocab), [0] * len(vocab)
    vocab_dic = {}
    for ind in range(len(vocab)):
        vocab_dic[vocab[ind]] = ind  # save the index

    tokens = context.split() #get the tokens
    for t in tokens:  # lowercase everything
        t = t.lower()
    headIndex = -1 #will be set to the index of the target word
    pKey, tKey, nKey = '', '', ''
    for i in range(len(tokens)):
        m = headMatch.match(tokens[i])
        if m: #a match: we are at the target token
            if i == 0 or i == len(tokens)-1:
                break  # not sure if this is right
            pWord = tokens[i-1]
            nWord = tokens[i+1]
            tokens[i] = m.groups()[0]
            headIndex = i
            tWord = tokens[i]  # now need to shave word to just the first '/'
            for j in range(len(tWord)):
                if tWord[j] == '/':
                    tWord = tWord[:j]
                    break
            tokens[i] = tWord

            if pWord in vocab_dic and nWord in vocab_dic:
                prevOneHot[vocab_dic[pWord]] = 1
                nextOneHot[vocab_dic[nWord]] = 1

    new_context = ' '.join(tokens)  # turn context back into string (optional)
    return prevOneHot, nextOneHot, new_context.lower()


def getFeatures(target_examples, vocab, senses):  # target example is the list of examples for a lemma
    features = []
    y = []
    for row in target_examples:
        pHot, nHot, new_context = removeHeadTagAndGetOneHot(row[2], vocab)
        pHot = np.array(pHot)
        nHot = np.array(nHot)
        features.append(np.concatenate((pHot, nHot)))
        y.append(senses[row[1]])
        row[2] = new_context

    return features, y


def CrossEntropyLoss1(ypred, ytrue):
    loss_function = nn.CrossEntropyLoss()
    output = loss_function(ypred, ytrue)
    return output


def getIndexesForTargetWord(target_examples):
    targetWordIndexes = {}
    for key in target_examples:
        if key not in targetWordIndexes:
            targetWordIndexes[key] = []
        for row in target_examples[key]:
            words = row[2].split(' ')
            for k in range(len(words)):
                if '<head>' in words[k]:
                    targetWordIndexes[key].append(k)
                    break

    return targetWordIndexes


def getEmbedding(word, embeddingDic):
    key = None
    try:
        key = embeddingDic[word]
    except KeyError:
        key = torch.zeros(50)
    return key


def getCooccuranceMatrix(target_examples, vocabIndex):
    cooccurrence_matrix = np.zeros((len(vocab) + 1, len(vocab) + 1))

    for key in target_examples:
        for row in target_examples[key]:
            words = row[2].split(' ')
            for i in range(0, len(words) - 1):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i], words[j]
                    ind1, ind2 = -1, -1
                    if w1 not in vocabIndex:
                        ind1 = len(vocab)
                    else:
                        ind1 = vocabIndex[w1]

                    if w2 not in vocabIndex:
                        ind2 = len(vocab)
                    else:
                        ind2 = vocabIndex[w2]

                    cooccurrence_matrix[ind1][ind2] += 1
                    cooccurrence_matrix[ind2][ind1] += 1

    return cooccurrence_matrix


def getFeaturesEmbeddings(target_examples, embeddingDic, target_word_indexes):
    newFeatures = []  # this will be X
    for key in target_examples:
        ind = 0
        firstRow = True
        temp_features = None
        for row in target_examples[key]:
            words = row[2].split(' ')
            t_word_index = target_word_indexes[key][ind]
            # print(words[t_word_index])
            p2 = words[t_word_index - 2] if t_word_index - 2 > -1 and t_word_index < len(words) else None
            p1 = words[t_word_index - 1] if t_word_index - 1 > -1 and t_word_index < len(words) else None
            n1 = words[t_word_index + 1] if t_word_index + 1 < len(words) else None
            n2 = words[t_word_index + 2] if t_word_index + 2 < len(words) else None

            e1 = getEmbedding(p2, embeddingDic)
            e2 = getEmbedding(p1, embeddingDic)
            e3 = getEmbedding(n1, embeddingDic)
            e4 = getEmbedding(n2, embeddingDic)

            if firstRow:
                temp_features = torch.cat((e1, e2, e3, e4))
                firstRow = False
            else:
                temp_features = torch.vstack((temp_features, torch.cat((e1, e2, e3, e4))))
            ind += 1
        temp_features = temp_features.type(torch.FloatTensor)
        newFeatures.append(temp_features)

    return newFeatures


## The Logistic Regression Class (do not edit but worth studying)
class LogReg(nn.Module):
    def __init__(self, num_feats, numClasses, learn_rate = 0.01, device = torch.device("cpu") ):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, numClasses) #add 1 to features for intercept

    def forward(self, X):
        #This is where the model itself is defined.
        #For logistic regression the model takes in X and returns
        #a probability (a value between 0 and 1)

        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        # return 1/(1 + torch.exp(-self.linear(newX))) #logistic function on the linear output
        return self.linear(newX)


###################################################################################
## MAIN


if __name__ == "__main__":
    ##DONT EDIT

    ##RUN PART 1: loading data, tokenize:
    print("\nLOADING DATA...")
    data, sensesOriginal, target_words = loadData("onesec_train.tsv")  # this function returns the data and target words and their unique ids
    removeLemmaPOS(data)

    word_freqs = {}
    for line in data:
        getWordFrequencies(line[2], word_freqs)  # get the counts

    get_top_2000_words(word_freqs)
    vocab = list(word_freqs.keys())

    # get just the target word examples to easily access 'process' , 'language', 'machine'
    target_examples = getTargetExamples(data)  # will return the X and y for each lemma
    target_word_indexes = getIndexesForTargetWord(target_examples)
    ids = getUniqueIds(target_examples)
    # making a list of all one hot encodings
    # extract features:
    features, senses = [], []
    i = 0
    for lemma in target_words:
        f, s = getFeatures(target_examples[lemma], vocab, ids[i])
        features.append(f)
        senses.append(s)
        i += 1

    Xs, ys = [], []
    for i in range(len(features)):
        x_ = torch.from_numpy(np.array(features[i]).astype(np.float32))
        y_ = torch.from_numpy(np.array(senses[i])).type(torch.LongTensor)
        Xs.append(x_)
        ys.append(y_)

    for i in range(len(Xs)):
        print("X shape: ", Xs[i].shape, ", y_process shape: ", ys[i].shape)
    #Model setup:
    learning_rate, epochs = 1.0, 300
    print("\nTraining Logistic Regression...")
    models, sgds = [], []
    for j in range(len(Xs)):
        m = LogReg(len(vocab)*2, len(ids[j]))  # fix this
        s = torch.optim.SGD(m.parameters(), lr=learning_rate)
        models.append(m)
        sgds.append(s)

    #training loop:
    for i in range(epochs):
        for j in range(len(models)):
            models[j].train()
            sgds[j].zero_grad()

            #forward pass
            ypred = models[j](Xs[j])
            loss = CrossEntropyLoss1(ypred, ys[j])

            #backward
            loss.backward()
            sgds[j].step()


            if i % 20 == 0:
                print("  epoch: %d, loss: %.5f" %(i, loss.item()))
    print("Done with training each model")
    #calculate accuracy on test set:
    testData, testSensesOriginal, testTargetWords = loadData('onesec_test.tsv')
    removeLemmaPOS(testData)

    testTargetExamples = getTargetExamples(testData)
    testTargetWordIndexes = getIndexesForTargetWord(testTargetExamples)

    testFeatures, testSenses = [], []
    i = 0
    for lemma in target_words:
        f, s = getFeatures(testTargetExamples[lemma], vocab, ids[i])
        testFeatures.append(f)
        testSenses.append(s)
        i += 1

    testXs, test_ys = [], []
    for j in range(len(testFeatures)):
        x_ = torch.from_numpy(np.array(testFeatures[j]).astype(np.float32))
        y_ = torch.from_numpy(np.array(testSenses[j])).type(torch.LongTensor)
        testXs.append(x_)
        test_ys.append(y_)

    print("\nAccuracy:")   ### SHOW 1 EXAMPLE FOR EACH WORD
    with torch.no_grad():
        for x in range(0, len(models)):
            ytestpred_prob = models[x](testXs[x])
            maxList = ytestpred_prob.argmax(axis=1)
            ytestpred_class = maxList
            #conitnue here
            print(target_words[x])
            print("correct: " + str((test_ys[x] == ytestpred_class).sum().item()) + " out of " + str(test_ys[x].shape[0]))
            if x == 0:
                print("predictions for process.NOUN.000018: " + str(ytestpred_prob[18]))
                print("predictions for process.NOUN.000024: " + str(ytestpred_prob[24]))
            if x == 1:
                print("predictions for machine.NOUN.000004: " + str(ytestpred_prob[4]))
                print("predictions for machine.NOUN.000008: " + str(ytestpred_prob[8]))
            if x == 2:
                print("predictions for language.NOUN.000008: " + str(ytestpred_prob[8]))
                print("predictions for language.NOUN.000014: " + str(ytestpred_prob[14]))

            print()
        # print("\nLogReg Model Test Set Accuracy: %.3f" % ((ytest == ytestpred_class).sum() / ytest.shape[0]))
        # print(  "Lexicon Test Set Accuracy:      %.3f" % ((ytest == np.array(lexPreds).T[0]).sum() / ytest.shape[0]))
    print("PART 1 DONE")

    # part 2 starts here
    # 2.1 - make a (vocab size+1) x (vocab size+1) co-occurrence matrix
    vocabIndex = {word: i for i, word in enumerate(vocab)}

    cooccurrence_matrix = getCooccuranceMatrix(target_examples, vocabIndex)
    # print(cooccurrence_matrix)
    cooccurrence_matrix = (cooccurrence_matrix - np.mean(cooccurrence_matrix)) / np.std(cooccurrence_matrix)  # why do i have 0 std?
    cooccurrence_matrix = torch.from_numpy(cooccurrence_matrix)  # we have tensor

    # print(cooccurrence_matrix)
    # 2.2 - Run PCA and extract static, 50 dimensional embeddings
    u, d, v = torch.svd(cooccurrence_matrix)
    u = u[:,:50]

    embeddingDic = {}
    for i in range(len(vocab)):
        embeddingDic[vocab[i]] = u[i]

    print("Distances for part 2.3:")
    print("('language', 'process'): " + str(np.linalg.norm(embeddingDic['language'] - embeddingDic['process'])))
    print("('machine', 'process'): " + str(np.linalg.norm(embeddingDic['machine'] - embeddingDic['process'])))
    print("('language', 'speak'): " + str(np.linalg.norm(embeddingDic['language'] - embeddingDic['speak'])))
    print("('word', 'words'):" + str(np.linalg.norm(embeddingDic['word'] - embeddingDic['words'])))
    print("('word', 'the'): " + str(np.linalg.norm(embeddingDic['word'] - embeddingDic['the'])) + "\n")

    # part 3.1
    newFeatures = getFeaturesEmbeddings(target_examples, embeddingDic, target_word_indexes)
    print("Part 3.1...")
    for i in range(len(newFeatures)):
        print("X shape: ", newFeatures[i].shape, ", y_process shape: ", ys[i].shape)

    newModels, newSgds = [], []
    epochs = 400
    print("\nTraining Logistic Regression 3.2...")
    for j in range(len(newFeatures)):
        m = LogReg(200, len(ids[j]))
        s = torch.optim.SGD(m.parameters(), lr=learning_rate)
        newModels.append(m)
        newSgds.append(s)

    # training loop:
    for i in range(epochs):
        for j in range(len(newModels)):
            newModels[j].train()
            newSgds[j].zero_grad()

            #forward pass
            ypred = newModels[j](newFeatures[j])
            loss = CrossEntropyLoss1(ypred, ys[j])

            #backward
            loss.backward()
            newSgds[j].step()


            if i % 20 == 0:
                print("  epoch: %d, loss: %.5f" %(i, loss.item()))
    print("Done with training each model")

    #3.3
    print("\nPart 3.3")
    newFeaturesTest = getFeaturesEmbeddings(testTargetExamples, embeddingDic, testTargetWordIndexes)

    print("\nAccuracy:")  # need to show example for each word here too
    with torch.no_grad():
        for x in range(0, len(newModels)):
            ytestpred_prob = newModels[x](newFeaturesTest[x])
            maxList = ytestpred_prob.argmax(axis=1)
            ytestpred_class = maxList
            print(target_words[x])
            print("correct: " + str((test_ys[x] == ytestpred_class).sum().item()) + " out of " + str(test_ys[x].shape[0]))
            if x == 0:
                print("predictions for process.NOUN.000018: " + str(ytestpred_prob[18]))
                print("predictions for process.NOUN.000024: " + str(ytestpred_prob[24]))
            if x == 1:
                print("predictions for machine.NOUN.000004: " + str(ytestpred_prob[4]))
                print("predictions for machine.NOUN.000008: " + str(ytestpred_prob[8]))
            if x == 2:
                print("predictions for language.NOUN.000008: " + str(ytestpred_prob[8]))
                print("predictions for language.NOUN.000014: " + str(ytestpred_prob[14]))
            print()

