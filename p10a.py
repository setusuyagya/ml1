import numpy as np
import matplotlib.pyplot as plt
def local_regression(x0,x,y,tau):
    x0=[1,x0]
    x=[[1,i] for i in x]
    x=np.array(x)
    xw=(x.T)*np.exp(np.sum((x-x0)**2,axis=1)/(-2*tau))
    beta=np.linalg.pinv(xw@x)@xw@y@x0
    return beta
def draw(tau):
    predictions=[local_regression(x0,x,y,tau) for x0 in domain]
    plt.plot(x,y,'o',color='blue')
    plt.plot(domain,predictions,color='red')
    plt.show()
x=np.linspace(-3,3,num=1000)
domain=x
y=np.log(np.abs(x**2-1)+0.5)
draw(100)
draw(0.1)