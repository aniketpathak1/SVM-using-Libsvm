# -*- coding: utf-8 -*-
"""
Created on Fri Nov 02 17:54:12 2018

@author: patha
"""

from svmutil import *
import sys
training_set = (sys.argv[1])
y, x = svm_read_problem(training_set)
prob = svm_problem(y,x)
param = svm_parameter('-s 0 -t 0 -d 3 -g 0.018 -r 0 -c 1 -n 0.5 -p 0.1 -m 100 -e 0.001 -h 1 -b 0 ')
m = svm_train(prob, param)
validation_set =(sys.argv[2])
y, x = svm_read_problem(validation_set)
p_label, p_acc, p_val = svm_predict(y,x, m)
ACC, MSE, SCC = evaluations(y, p_label)
print(ACC)
print(MSE)
print(SCC)

