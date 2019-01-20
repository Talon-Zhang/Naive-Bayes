# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import numpy as np

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
    Then train_set := [['i','like','pie'], ['i','like','cake']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was spam and second one was ham.
    Then train_labels := [0,1] typo:[1,0]

    ************1 is spam and 0 is ham.************

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # return predicted labels of development set

    #pre-filter the unimportant words in our emails
    remove_list = [',', '.', '@', ';', 'the', 'and', 'or', 'a', '/', '-', '(', ')']
    for email in train_set:
        for word in email:
            for remove_word in remove_list:
                if word == remove_word:
                    email.remove(word)
    for email in dev_set:
        for word in email:
            for remove_word in remove_list:
                if word == remove_word:
                    email.remove(word)

    ind_spam = list(np.where(np.asarray(train_labels) == 1)[0])
    ind_ham = list(np.where(np.asarray(train_labels) == 0)[0])
    # list storing the spam and ham in the train set
    spam_list = [train_set[i] for i in ind_spam]
    ham_list = [train_set[i] for i in ind_ham]

    # total number of words in spam training set and ham training set
    spam_word_count = 0
    ham_word_count = 0
    for l in spam_list:
        spam_word_count += len(l)
    for l in ham_list:
        ham_word_count += len(l)

    # dictionary storing np.log(P(word|spam)). Key is the word in string
    spam_dict = {}
    # dictionary storing np.log(P(word|ham)). Key is the word in string
    ham_dict = {}

    # storing the (word count + smoothing_parameter) in the dict for each word
    for l in spam_list:
        for w in l:
            if w in spam_dict:
                spam_dict[w] += 1
            else:
                spam_dict[w] = 1+smoothing_parameter

    for l in ham_list:
        for w in l:
            if w in ham_dict:
                ham_dict[w] += 1
            else:
                ham_dict[w] = 1+smoothing_parameter

    # consider the laplace smoothing
    spam_dict['UNK'] = smoothing_parameter
    ham_dict['UNK'] = smoothing_parameter

    # now compute P(word|type) = 
    # (word count + smoothing_parameter)/(tot # of words + smoothing_parameter*(# of different words + 1))
    # take the log and store in the dict
    denominator_spam = spam_word_count+smoothing_parameter*(len(spam_dict)+1)
    denominator_ham = ham_word_count+smoothing_parameter*(len(ham_dict)+1)

    for key in spam_dict.keys():
        n = spam_dict[key]
        spam_dict[key] = np.log(n/denominator_spam)
    for key in ham_dict.keys():
        n = ham_dict[key]
        ham_dict[key] = np.log(n/denominator_ham)

    # computing the probability for the dev_set. calculate both P(words|ham) and P(words|spam), and compare which
    # is larger
    dev_label = []
    for email in dev_set:
        P_spam = 0
        P_ham = 0
        for word in email:
            if word in spam_dict.keys():
                P_spam += spam_dict[word]
            else:
                P_spam += spam_dict['UNK']

            if word in ham_dict.keys():
                P_ham += ham_dict[word]
            else:
                P_ham += ham_dict['UNK']
        if P_spam > P_ham:
            dev_label.append(1)
        else:
            dev_label.append(0)

    return dev_label


######################################## Extra Credit ##############################################


# return a list of corresponding (P_spam, P_ham) for each email in dev_set
def probability(spam_list, ham_list, dev_set, smoothing_parameter):
    # total number of words in spam training set and ham training set
    spam_word_count = 0
    ham_word_count = 0
    for l in spam_list:
        spam_word_count += len(l)
    for l in ham_list:
        ham_word_count += len(l)

    # dictionary storing np.log(P(word|spam)). Key is the word in string
    spam_dict = {}
    # dictionary storing np.log(P(word|ham)). Key is the word in string
    ham_dict = {}

    # storing the (word count + smoothing_parameter) in the dict for each word
    for l in spam_list:
        for w in l:
            if w in spam_dict:
                spam_dict[w] += 1
            else:
                spam_dict[w] = 1 + smoothing_parameter

    for l in ham_list:
        for w in l:
            if w in ham_dict:
                ham_dict[w] += 1
            else:
                ham_dict[w] = 1 + smoothing_parameter

    # consider the laplace smoothing
    spam_dict['UNK'] = smoothing_parameter
    ham_dict['UNK'] = smoothing_parameter

    # now compute P(word|type) =
    # (word count + smoothing_parameter)/(tot # of words + smoothing_parameter*(# of different words + 1))
    # take the log and store in the dict
    denominator_spam = spam_word_count + smoothing_parameter * (len(spam_dict) + 1)
    denominator_ham = ham_word_count + smoothing_parameter * (len(ham_dict) + 1)

    for key in spam_dict.keys():
        n = spam_dict[key]
        spam_dict[key] = np.log(n / denominator_spam)
    for key in ham_dict.keys():
        n = ham_dict[key]
        ham_dict[key] = np.log(n / denominator_ham)

    # computing the probability for the dev_set. calculate both P(words|ham) and P(words|spam)
    # and return a tuple of them
    dev_label = []
    for email in dev_set:
        P_spam = 0
        P_ham = 0
        for word in email:
            if word in spam_dict.keys():
                P_spam += spam_dict[word]
            else:
                P_spam += spam_dict['UNK']

            if word in ham_dict.keys():
                P_ham += ham_dict[word]
            else:
                P_ham += ham_dict['UNK']
        dev_label.append((P_spam, P_ham))

    return dev_label


def naiveBayes_ec(train_set, train_labels, dev_set, smoothing_parameter):
    # return predicted labels of development set
    lamda = 0

    # pre-filter the unimportant words in our emails
    remove_list = [',', '.', '@', ';', 'the', 'and', 'or', 'a', '/', '-', '(', ')']
    for email in train_set:
        for word in email:
            for remove_word in remove_list:
                if word == remove_word:
                    email.remove(word)
    for email in dev_set:
        for word in email:
            for remove_word in remove_list:
                if word == remove_word:
                    email.remove(word)

    # find the unigram probability
    ind_spam = list(np.where(np.asarray(train_labels) == 1)[0])
    ind_ham = list(np.where(np.asarray(train_labels) == 0)[0])
    # list storing the spam and ham in the train set
    spam_list = [train_set[i] for i in ind_spam]
    ham_list = [train_set[i] for i in ind_ham]
    unigram_probabilities = probability(spam_list, ham_list, dev_set, smoothing_parameter)

    # find the bigram probability
    bi_spam = []
    bi_ham = []
    bi_dev = []
    for email in spam_list:
        bi_email = []
        for i in range(len(email)-1):
            bi_email.append((email[i], email[i+1]))
        bi_spam.append(bi_email)
    for email in ham_list:
        bi_email = []
        for i in range(len(email)-1):
            bi_email.append((email[i], email[i+1]))
        bi_ham.append(bi_email)
    for email in dev_set:
        bi_email = []
        for i in range(len(email) - 1):
            bi_email.append((email[i], email[i + 1]))
        bi_dev.append(bi_email)
    bigram_probabilities = probability(bi_spam, bi_ham, bi_dev, smoothing_parameter)

    dev_label = []
    for i in range(len(dev_set)):
        P_unigram_spam = unigram_probabilities[i][0]
        P_unigram_ham = unigram_probabilities[i][1]
        P_bigram_spam = bigram_probabilities[i][0]
        P_bigram_ham = bigram_probabilities[i][1]

        P_spam = (1-lamda)*P_unigram_spam + lamda*P_bigram_spam
        P_ham = (1-lamda)*P_unigram_ham + lamda*P_bigram_ham

        if P_spam > P_ham:
            dev_label.append(1)
        else:
            dev_label.append(0)

    return dev_label

