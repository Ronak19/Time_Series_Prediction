import xlrd
import matplotlib.pyplot as plt
import numpy as np
import random as rd

data = xlrd.open_workbook("project3_time series data_students.xlsx")  #training data set uploaded
sheet = data.sheet_by_index(0)
label_val1 = sheet.col_values(0)
trainset_fraction = 1
label_n = label_val1[:int(np.floor(trainset_fraction*len(label_val1)))]
scale = max(label_n)
label_val = [label_n[i]/scale for i in range(len(label_n))]
len_H1 = 17
len_input = 6

theta1 = [[rd.uniform(0, 0.5) for i in range(len_H1)] for j in range(len_input)]
theta2 = [rd.uniform(0, 0.5) for i in range(len_H1)]
eta = 0.1
tolerence = 0.00001

def g(x):
    return (1/(1 + np.exp(-x)))

c = [0.5]
cnt = 0
ouput = [0.0 for b in range(len(label_val) - len_input)]
while(c[cnt] > tolerence):
    for k in range(len(label_val) - len_input):
        if (c[cnt] > tolerence):
            z = [0.0 for b in range(len_H1)]
            a = [0.0 for b in range(len_H1)]
            d_0 = [0.0 for b in range(len_H1)]
            #Forward Propagation
            for i in range(len_H1):
                for j in range(len_input):
                    z[i] += label_val[k+len_input-j-1] * theta1[j][i]
            for i in range(len_H1):
                a[i] = g(z[i])
            for i in range(len_H1):
                ouput[k] += theta2[i] * a[i]
            ouput[k] = g(ouput[k])
            #Back Propagation
            c.append(0.5 * np.square(ouput[k] - label_val[k+len_input]))
            d = (ouput[k] - label_val[k+len_input]) * ouput[k] * (1 - ouput[k])
            for i in range(len_H1):
                d_0[i] = theta2[i] * d * a[i] * (1 - a[i])
            for i in range(len_H1):
                theta2[i] -= eta * a[i] * d
            for i in range(len_input):
                for j in range(len_H1):
                    theta1[i][j] -= eta * label_val[k+len_input-i-1] * d_0[j]
            cnt += 1
        else:
            break

x = list(range(len(c)))
plt.plot(x, c)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')
plt.title('Loss curve')
plt.show()

#test Accuracy
label_n = label_val1[int(np.ceil((0.11)*len(label_val1))):]
label_val = [label_n[i]/scale for i in range(len(label_n))]
ouput = [0.0 for b in range(len(label_val))]
for i in range(len_input):
    ouput[i] = label_val[i]
for k in range(len(label_val) - len_input):
    z = [0.0 for b in range(len_H1)]
    a = [0.0 for b in range(len_H1)]
    d_0 = [0.0 for b in range(len_H1)]
    for i in range(len_H1):
        for j in range(len_input):
            z[i] += ouput[k+len_input-j-1] * theta1[j][i]
    for i in range(len_H1):
        a[i] = g(z[i])
    for i in range(len_H1):
        ouput[k+len_input] += theta2[i] * a[i]
    ouput[k+len_input] = g(ouput[k+len_input])

for i in range(len(ouput)):
    ouput[i] *= scale
ms = 0
for i in range(len(ouput) - len_input):
    ms += np.square(label_n[i+len_input] - ouput[i+len_input])
ms /= len(label_n)
ms = np.sqrt(ms)
print('RMSE for test = ', ms)

#Prediction
ouput = [0.0 for b in range(30)]
for i in range(len_input):
    ouput[i] = label_val1[len(label_val1)-len_input+i]/scale
for k in range(len(ouput) - len_input):
    z = [0.0 for b in range(len_H1)]
    a = [0.0 for b in range(len_H1)]
    d_0 = [0.0 for b in range(len_H1)]
    for i in range(len_H1):
        for j in range(len_input):
            z[i] += ouput[k+len_input-j-1] * theta1[j][i]
    for i in range(len_H1):
        a[i] = g(z[i])
    for i in range(len_H1):
        ouput[k+len_input] += theta2[i] * a[i]
    ouput[k+len_input] = g(ouput[k+len_input])

for i in range(len(ouput)):
    ouput[i] *= scale
print('Predicted values = ', ouput[len_input:])

#Test RMSE
testfile = input('Test File name = ')      #load the test data points xlsx file here
data1 = xlrd.open_workbook(testfile)
sheet1 = data1.sheet_by_index(0)
test_data = sheet1.col_values(0)
ms = 0
for i in range(len(ouput) - len_input):
    ms += np.square(test_data[i] - ouput[i+len_input])
ms /= len(test_data)
ms = np.sqrt(ms)
print('RMSE for test = ', ms)
