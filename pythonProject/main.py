import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split


#def max_value(inputlist):
#    return max([sublist[-1] for sublist in inputlist])

#def min_value(inputlist):
#    return min([sublist[-1] for sublist in inputlist])

"""
def normalize(x):
    a = 0
    b = 1
    j = 0
    xmax = int(max_value(x))
    xmin = int(min_value(x))
    print("X max ", xmax)
    print("X min ", xmin)
    for i in x:
        #print(i[1])
        x[j] = int((i - xmin)/(xmax - xmin))
        j += 1
    #print("X = ", X)
    print("x = ", x)
    return x
"""


data = pd.read_csv('Development Index.csv')
#print(data)
data = data.transpose()
data = data.values.tolist()
data = np.array(data)

#data = data.values.tolist()
#print("data", data)

#print(data[["Population", "GDP ($ per capita)", "Pop. Density "]])


stat, p = shapiro(data[0])
print("stat = ", stat, "pvalue = ", p)
shapiro_test = stats.shapiro(data)


x = [data[0], data[2], data[3]]
y = data[6]
y_norm = []
#y = np.array(y)
#yMax = int(max(data[6]))


#xmax = int(max(max(data[0]), max(data[2]), max(data[3])))
#print("XMAX ", xmax)
#print("x = ", x)
#print("Y ", y)

#print("y =", y)
#y = np.dot(x, np.array([0, 224]))
#print("y = ", y)
#xNorm = normalize(data[["Population"]])
#print(xNorm)



#xNorm = preprocessing.minmax_scale(x)

x = np.array(x)
x = x.transpose()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


for i in range(len(y_train)):
    y_norm.append(y_train[i] / (4 + (4 - 1)))
    i += 1
print("y normalize: ", y_norm)
xtest_denorm = x_test


#yNorm = preprocessing.normalize(y)
#yNorm = yNorm.transpose()

xtrain_norm = preprocessing.normalize(x_train, norm="max", axis=0)
xtest_norm = preprocessing.normalize(x_test, norm="max", axis=0)
#print("xtrainNorm: ", xtrain_norm, "xtestNorm: ", xtest_norm)
#x_norm = x_norm.transpose()
#print("NORM ", xNorm)


linear = LinearRegression()
reg = linear.fit(xtrain_norm, y_norm)
pre = linear.predict(xtest_norm)

#preNorm = preprocessing.normalize(pre)
#preNorm.transpose()
#print(pre)


print("liner_coef_: ", linear.coef_)
print("liner_inter: ", linear.intercept_)


"""
y_pre = np.array(pre)
pca = decomposition.PCA()
pca.fit(y_pre)
pca.fit(y_true)
y_true = pca.transform(y_true)
"""#///денормализация

#for i in y:
#    y_true.append(int(i))
#for i in y:
#    y_pre.append(int(i))

#print("y_true", y_true)
#print("y_pre", y_pre)

y_pre = []
y_true = y_norm
a = b = c = 0

for k in range(len(y_true)):
    y_true[a] = (y_true[k] * (4 + (4 - 1)))
    a += 1


for k in range(len(pre)):
    y_pre.append(pre[k] * (4 + (4 - 1)))
    b += 1

print("lenxtest: ", len(x_test))
for k in range(len(x_test)):
    x_test[c] = (x_test[k] * (4 + (4 - 1)))
    c += 1


print("y true: ", y_true)
print("y pre: ", y_pre)


RMSE_Error = mean_squared_error(y_test, y_pre)
print("RMSE_Error = ", RMSE_Error)

y_t = y_test
y_test = np.resize(y_test, 20)
y_pre = np.resize(y_pre, 20)

#xtest_d = np.resize(xtest_denorm, 20)

dt = {"y_t": y_test, "y_p": y_pre}
df = pd.DataFrame(dt)
df.plot(kind = "bar")
plt.show()


#print("yt: ", y_t)
#print("x_test: ", xtest_denorm[:, 1])


plt.scatter(xtest_denorm[:, 1], y_t)
plt.plot(xtest_denorm[:, 1], (linear.coef_[1] * xtest_denorm[:, 1]) + linear.intercept_, color='red', linewidth=2)
plt.show()


#plt.hist([y_pre, y_true], bins)
#plt.hist(y_pre, bins, alpha=0.5, label='x')
#plt.hist(y_true, bins, alpha=0.5, label='y')

#plt.show()


#plt.hist(y_true, alpha = 0.5, color="blue")
#plt.hist(y_pre, alpha = 0.5, color="orange")

#plt.show()


#print(shapiro_test)
#print(stat, p)



#with open('Development Index.csv', mode='r') as File:
    #fieldNames = ["Population", "Pop. Density", "GDP ($ per capita)"]
    #data = csv.DictReader(File, fieldNames)
    #for row in data:
        #print(row["Population"], row["Pop. Density"], row["GDP ($ per capita)"])

#print(data)

#print(data.reader)
#p = shapiro(data["Population"])
#print(stat)
#print(p)


#plt.hist(row["GDP ($ per capita)"])

#plt.show()