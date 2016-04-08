#!/usr/bin/env python3

# Random paring and one-point crossover continuous GA


from scipy.integrate import odeint
from pylab import *
from math import exp,log,erfc,sqrt
import matplotlib.pyplot as plt
from random import random, sample, uniform, shuffle
import time
from scipy.stats import chisquare
from statistics import variance
import multiprocessing as mp
import pandas as pd
import numpy as np
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import roc_auc_score
from sklearn.externals import six
from seqlearn._utils import atleast2d_or_csr, safe_sparse_dot, validate_lengths
from tqdm import tqdm

def predict(self, X, lengths=None):
    X = atleast2d_or_csr(X)
    scores = safe_sparse_dot(X, self.coef_.T)
    if hasattr(self, "coef_trans_"):
        n_classes = len(self.classes_)
        coef_t = self.coef_trans_.T.reshape(-1, self.coef_trans_.shape[-1])
        trans_scores = safe_sparse_dot(X, coef_t.T)
        trans_scores = trans_scores.reshape(-1, n_classes, n_classes)
    else:
        trans_scores = None
        
    decode = self._get_decoder()

    if lengths is None:
        y = decode(scores, trans_scores, self.intercept_trans_,
                   self.intercept_init_, self.intercept_final_)
    else:
        start, end = validate_lengths(X.shape[0], lengths)
    
        y = [decode(scores[start[i]:end[i]], trans_scores,
                    self.intercept_trans_, self.intercept_init_,
                    self.intercept_final_)
            for i in six.moves.xrange(len(lengths))]
        y = np.hstack(y)
    
    return self.classes_[y], scores


def To_AUC_label(labels):
    labels_new = []    
    for label in labels:
        if label < 1:
            labels_new.append(1)
        else:
            labels_new.append(0)

    return labels_new

def set_slice(df, time):
    df = df.sort_index(level=0)
    df = df.loc[time:]
    df = df.sort_index(level=1)
    
    return df
    
def length(df):
    df_length = df.count(level=1).ix[:,0]
    df_length = df_length.replace(to_replace=0, value=np.nan)
    df_length = df_length.dropna()
    length_list = list(df_length)
    
    return length_list



def simulation(individual):

    clf = StructuredPerceptron(lr_exponent=0.01, max_iter=100, random_state=2)
    clf.fit(x_train, y_train, lengths)
    
    pred, pred_scores = predict(clf, x_test, lengths_test)
    
    test_labels_new = To_AUC_label(y_true)    
    pred_labels_new = To_AUC_label(pred)
        
    return (test_labels_new,pred_labels_new)


##############    GA    #################

def population(Nind, imin, imax):
    # Set up initial population matrix
    pOR = list([] for _ in range(2))
    for z in range(2):
        pOR[z] = [uniform(imin[z], imax[z])]
        while len(pOR[z]) < Nind:
            iOR = uniform(imin[z], imax[z])
            pOR[z].append(iOR)
            for c in range(len(pOR[z])-1):
                if iOR == pOR[z][c]:
                    pOR[z].pop()
                    break
    
    populationall = list(zip(*pOR))
    return populationall


def fitness(individual):

    test_labels_new, pred_labels_new = simulation(individual)

    ###### Cost function

    cost = 1-roc_auc_score(test_labels_new, pred_labels_new)
    
    return cost


def tuple_of_population(listout,populationall):
    # Build a tuple of list with a chromosome and a cost value
    # in every element of tuple
    poptup = list(zip(listout, populationall))
    #print(poptup)
    return poptup


def selection(poptup, Nkeep):
    parents = []
    # Elite
    for candidate in range(2):
        parents.append(list(sorted(poptup)[candidate][1]))
    #Tournament selection
    while len(parents) < Nkeep:
        parents.append(list(sorted(sample(poptup, 3))[0][1]))
    return parents


def mating(parents, Nind, Nkeep):
    while len(parents) < Nind:
        # Random pairing
        famo = sample(range(0, Nkeep), 2)
        fa = parents[famo[0]]
        mo = parents[famo[1]]
        # The blending method
        offspring1 = []
        offspring2 = []
        for x in range(2):
            beta = random()
            offspring1.append(float(mo[x])*beta + float(fa[x])*(1-beta))
            offspring2.append(float(fa[x])*beta + float(mo[x])*(1-beta))
        parents.append(offspring1)
        parents.append(offspring2)
        if len(parents) > Nind:
            parents.pop()
    return parents


def mutation(u, parents, Nkeep, imin, imax):
    for individual in parents[Nkeep:]:
        for cond in range(2):
            if u > random():
                individual[cond] = uniform(imin[cond], imax[cond])

    return parents


def average(poptup):
    total = 0
    for i in range(len(poptup)):
        total += poptup[i][0]
        average = total/len(poptup)
    return average


######## MULTIPROCESS #########

def multiprocess(processes, processfunction, argurerange):
    pool = mp.Pool(processes = processes)
    outputs = [pool.apply_async(processfunction, args=(something,)) for something in argurerange]
    results = [p.get() for p in outputs]
    pool.terminate()
    return results


############# MAIN #############

def iterationmain(subs):
    popl = subs
    Nkeep = int(Xrate * len(popl))
    eachopt = list(fitness(individual) for individual in popl)
    #eachopt = multiprocess(2, fitness, popl)
    subs = list(mutation(u, mating(selection(tuple_of_population(eachopt, popl), Nkeep), Nind, Nkeep), Nkeep, imin, imax))
    history = sorted(eachopt).pop(0)
    solution = list(sorted(list(zip(eachopt, subs)))[0][1])

    return [subs,history,solution]

for i in range(1):
    imin = [0.001, 100]
    imax = [0.01, 1000]
    Nindi = 10
    Nind = 75
    Xrate = 0.5
    u = 0.04

    ax = []

    start_time = time.time()
    print(start_time)
    
    for i in range(1):
    
        df_train = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/training_data_'+str(i+27)+'.pickle')
        df_train_label = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/training_labels_'+str(i+27)+'.pickle')
        df_test = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/test_data_'+str(i+27)+'.pickle')
        df_test_label = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/test_labels_'+str(i+27)+'.pickle')
        
        x_train = set_slice(df_train, '20140901')
        y_train = set_slice(df_train_label, '20140901')
        
        x_test = df_test.sort_index(level=0)
        y_test = df_test_label.sort_index(level=0)
        
        lengths = length(x_train)
        lengths_test = length(x_test)
        
        y_true = np.asarray(list(y_test))    
    
    
    
    popl = population(Nindi, imin, imax)
    Nkeep = int(Xrate * len(popl))

    history = []
    solution = []
    print(i,"--", 0, "--", time.time() - start_time)
    for gen in range(2):
        
        popl,history_og,solution_og = iterationmain(popl)
        history.append(history_og)
        solution.append(solution_og)
        print(i,"--", gen+1, "--", time.time() - start_time)
    
    
    
    
    
    '''
    popl = population(Nindi, imin, imax)
    Nkeep = int(Xrate * len(popl))
    generation = 0
    eachopt = multiprocess(2, fitness, popl)
    '''
    # print(eachopt)
    # history = [sorted(eachopt).pop(0)]
    # print(history)
    # mean = [average(tuple_of_population(eachopt, popl))]

    '''  

    subp = list(highcand[1] for highcand in list(sorted(list(zip(eachopt, popl)))[:300]))
    shuffle(subp)
    subp = list(map(list, subp))
    sub0 = subp[0:75]
    sub1 = subp[75:150]
    sub2 = subp[150:225]
    sub3 = subp[225:300]
    '''
    
'''       
    for generation in tqdm(range(2)):
        if generation%20 == 19:
            shuffle(sub0)
            shuffle(sub1)
            shuffle(sub2)
            shuffle(sub3)
            for elepos in range(2):
                sub0.append(sub3.pop(elepos))
                sub1.append(sub0.pop(elepos))
                sub2.append(sub1.pop(elepos))
                sub3.append(sub2.pop(elepos))
'''
        # variances = [] 
        
'''
        results = multiprocess(2, iterationmain,[sub0, sub1, sub2, sub3])

        sub0 = results[0][0]
        history[0].append(results[0][1])
        solution[0] = results[0][2]

        sub1 = results[1][0]
        history[1].append(results[1][1])
        solution[1] = results[1][2]

        sub2 = results[2][0]
        history[2].append(results[2][1])
        solution[2] = results[2][2]

        sub3 = results[3][0]
        history[3].append(results[3][1])
        solution[3] = results[3][2]
        
        gen = generation + 1
        ax.append(gen)
        print(i,"--", gen, "--", time.time() - start_time)

'''
    ################################
'''
    for solnum in range(4):
        print(solution[solnum])
    
    plt.figure(0)
    plt.subplot(4,1,1)
    plt.plot(ax, history[0])
    plt.ylabel('Minimum')

    plt.subplot(4,1,2)
    plt.plot(ax, history[1])
    plt.ylabel('Minimum')

    plt.subplot(4,1,3)
    plt.plot(ax, history[2])
    plt.ylabel('Minimum')

    plt.subplot(4,1,4)
    plt.plot(ax, history[3])
    plt.ylabel('Minimum')
    plt.xlabel('generation')

    del popl
    del eachopt
    del history
    del solution

# plt.axis([0, 100, -50, 50])
plt.show()
'''