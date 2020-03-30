# -*- coding: utf-8 -*-
"""
Created on Thu Nov 01 12:40:28 2018

@author: Jie.Hu
"""

import numpy as np
from scipy.optimize import minimize

a = np.array([1.2, 2.3, 4.2])
b = np.array([1, 5, 6])
c = np.array([5.4, 6.2, 1.9])

m = np.vstack([a,b,c])
y = np.array([5.3, 0.9, 5.6])

def loss(x):
    return np.sum(np.square((np.dot(x, m) - y)))

def loss_mape(x):
    return np.mean(np.abs((np.dot(x, m) - y))/y)

cons = ({'type': 'eq',
         'fun' : lambda x: np.sum(x) - 1.0})

x0 = np.zeros(m.shape[0])
res = minimize(loss_mape, x0, method='SLSQP', constraints=cons,
               bounds=[(0, np.inf) for i in range(m.shape[0])], options={'disp': True})

print(res.x)
print(np.dot(res.x, m.T))
print(np.sum(np.square(np.dot(res.x, m) - y)))


def custom_mape(true, pred):
    return np.mean(np.abs(pred - true) / true)

custom_mape(y, a) 