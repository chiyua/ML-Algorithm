import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

#Load data
x_data=np.random.rand(50) #feature
print("x_data:",x_data)
y_data=np.random.randint(1,100,size=(50)) #label
plt.scatter(x_data,y_data)
plt.show()

#learning rate
lr=0.1
#intercept
b=0
#slope
k=0
#epoches
epoches=20

#least squares method
def compu_meanError(k,b,x_data,y_data):
    total_error=0
    for i in range(0,len(x_data)):
        total_error+=(y_data[i]-(k*x_data[i]+b))**2
    mean_error=total_error/float(len(x_data))
    return mean_error

def gradient_decent_runner(lr,k,b,epoches,x_data,y_data):
    #Total number of data
    total_num=float(len(x_data))
    #Loop epoches
    for i in range(0,epoches):
        k_grad=0 
        b_grad=0
        for j in range (0, len(x_data)):
            k_grad+=-(1/(2.0*total_num))*(y_data[j]-(k*x_data[j]+b))
            b_grad+=-(1/(2.0*total_num))*x_data[j]*(y_data[j]-(k*x_data[j]+b))
        #update k and b
        b=b-(lr*b_grad)
        k=k-(lr*k_grad)
        if i%5==0:
            plt.scatter(x_data,y_data)
            plt.plot(x_data,k*x_data+b,'r')
            plt.show()
    return b,k

print("Starting b={0},k={1}, error={2}".format(b,k,compu_meanError(b,k,x_data,y_data)))
print("running...")
b,k=gradient_decent_runner(lr,k,b,epoches,x_data,y_data)
print("After {0} iterations b={1}, k={2}, error={3}".format(epoches,b,k,compu_meanError(b,k,x_data,y_data)))
plt.scatter(x_data,y_data)
plt.plot(x_data,k*x_data+b,'r')
plt.show()


