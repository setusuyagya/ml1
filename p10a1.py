# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:17:12 2020

@author: Setu Suyagya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def local_regression(a,b,c,tau):
    a=[1,a]
    b=[[1,i] for i in b]
    b=np.array(b)
    xw=(b.T)*np.exp(np.sum((b-a)**2,axis=1)/(-2*tau))
    beta=np.linalg.pinv(xw@b)@xw@c@a
    return beta
def draw(tau):
    predictions=[local_regression(x0,x,y,tau) for x0 in domain]
    plt.plot(x,y,'o',color='blue')
    plt.plot(domain,predictions,color='red')
    plt.show()
x=np.linspace(-3,3,num=1000)
domain=x
y=np.log(np.abs(x**2-1)+0.5)
draw(1000777)
draw(01.1)