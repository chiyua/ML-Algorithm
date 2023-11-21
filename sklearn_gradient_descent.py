from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

#load data
x_data=np.random.rand(20,1)
y_data=np.random.randint(1,100,size=(20,1))
print("x_data{0},y_data{1}".format(x_data,y_data))
plt.scatter(x_data,y_data)
plt.show()

#build model
model=LinearRegression()
#fit model
model.fit(x_data,y_data)

plt.scatter(x_data,y_data)
plt.plot(x_data,model.predict(x_data),'r')
plt.show()

