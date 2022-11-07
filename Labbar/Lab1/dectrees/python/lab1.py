# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:57:53 2022

@author: henri
"""


import matplotlib.pyplot as plt
import monkdata as m
import dtree as d
import numpy as np
import drawtree_qt5 as drawtree
import random

"""
Assignment 1:
"""

monkarr = [m.monk1, m.monk2, m.monk3]

e = np.zeros(3)
for i in range(len(e)):
    e[i] = d.entropy(monkarr[i])


print('Entropy(MONK-1): ', e[0], '\nEntropy(MONK-2): ', e[1], '\nEntropy(MONK-3): ', e[2])

"""
Assignment 3:
"""

i = 0
j = 0
a = np.zeros(shape=(len(monkarr),6))
for monk in monkarr:
    for j in range(6):
        a[i,j] = d.averageGain(monk, m.attributes[j])
        
    i = i+1


print(a)


"""
Midsection
"""



"""
Assignment 5:
"""


tree1 = d.buildTree(monkarr[0], m.attributes)
print(d.check(tree1, m.monk1))
print(d.check(tree1, m.monk1test))

tree2 = d.buildTree(monkarr[1], m.attributes)
print(d.check(tree2, m.monk2))
print(d.check(tree2, m.monk2test))

tree3 = d.buildTree(monkarr[2], m.attributes)
print(d.check(tree3, m.monk3))
print(d.check(tree3, m.monk3test))


#drawtree.drawTree(tree1)



"""
Assignment 7:
"""


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]
    
    

def proon(fraction,dataset):
    Bool = True
    cap = 0
    monk1train, monk1val = partition(dataset, fraction)
    t = d.buildTree(monk1train, m.attributes);
    BestPrune = 999999999 #it's over 9000!
    while Bool == True:

        if cap == 0:
            Pruned = d.allPruned(t)
        else:
            Pruned = d.allPruned(BestPrune)
        val = 0
        for prune in Pruned:
            if d.check(prune, monk1val)>val:
                val = d.check(prune, monk1val)
                best = val
                BestPrune = prune

        #print("best",best)
        if (cap>best):
            break

        cap = best
    return cap



Fractions = [i/10 for i in range(3,9)]
runs = 200
performance = np.zeros(shape=(1,6))
bad_performance = np.zeros(shape=(1,6))

k = 0
for fraction in Fractions:
    i = 0
    errorsum = 0
    bad_errorsum = 0
    while runs>i:
        
        bad_train, bad_test = partition(m.monk3, fraction)
        badtree = d.buildTree(bad_train, m.attributes)
        bad_accuracy = d.check(badtree, bad_test)
        
        bad_error = 1-bad_accuracy
        bad_errorsum = bad_errorsum + bad_error
        
        accuracy = proon(fraction,m.monk3)
        error = 1-accuracy
        errorsum = errorsum+error
        i = i+1

    averror = errorsum/i
    performance[0,k] = averror
    
    bad_averror = bad_errorsum/i
    bad_performance[0,k] = bad_averror
    
    k+=1

plt.plot(Fractions,list(np.transpose(performance)), label = 'Pruning')
plt.plot(Fractions,list(np.transpose(performance)), "bo")
plt.plot(Fractions,list(np.transpose(bad_performance)), label = 'Non-Pruning')
plt.plot(Fractions,list(np.transpose(bad_performance)), "bo")
plt.xlabel("Fractions")
plt.ylabel("Average Error")
plt.legend(loc="upper right")

print(performance)
    
    
    
    