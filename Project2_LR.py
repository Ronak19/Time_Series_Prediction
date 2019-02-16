import xlrd
import matplotlib.pyplot as plt
import numpy as np

data = xlrd.open_workbook("project2_time series data_students.xlsx")  #training data set uploaded
sheet = data.sheet_by_index(0)
label_val = sheet.col_values(0)
x1 = []
x2 = []
scale = max(label_val)
m = 275
order1 = 1
order2 = 1
theta1 = 0.8
theta2 = 0.8
alpha = 0.1
theta1_old = 0.0
theta2_old = 0.0
for k in range(2, m):
    x1.append(label_val[k-2])
for k in range(2, m):
    x2.append(label_val[k-1])

j = []
while ((theta1 < theta1_old - 0.00001 or theta1 > theta1_old + 0.00001) \
            and (theta2 < theta2_old - 0.00001 or theta2 > theta2_old + 0.00001)):
    err_sum1 = 0.0
    err_sum2 = 0.0
    err_fn = 0.0
    theta1_old = theta1
    theta2_old = theta2
    for i in range(2, m-2):
        err_fn = err_fn + np.power((theta1 * np.power(x1[i]/scale, order1) + theta2 * np.power(x2[i]/scale, order2) - label_val[i]/scale),2)
        err_sum1 = err_sum1 + (theta1 * np.power(x1[i]/scale, order1) + theta2 * np.power(x2[i]/scale, order2) - label_val[i]/scale) * (np.power(x1[i]/scale, order1))
        err_sum2 = err_sum2 + (theta1 * np.power(x1[i]/scale, order1) + theta2 * np.power(x2[i]/scale, order2) - label_val[i]/scale) * (np.power(x2[i]/scale, order2))
    j.append((1/(2*m)) * err_fn)
    theta1 = theta1 - alpha * (1/m) * err_sum1
    theta2 = theta2 - alpha * (1/m) * err_sum2

#plot cost function and regression model
plt.figure(1)
plt.plot(j)
plt.xlabel('No. of iterations')
plt.ylabel('Cost function (J)')
plt.title('Cost function vs No. of iterations')
x = [i+2 for i in range(m-2)]
print('Actual data points = ', x)
y = [(theta1 * np.power(x1[i-2]/scale, order1) + theta2 * np.power(x2[i-2]/scale, order2)) for i in x]
y1 = []
for i in y:
    y1.append(i*scale)
print('Predicted values for training data = ', y1)

plt.figure(2)
plt.plot(x, label_val[2:m], 'ro', x, y1)
plt.legend(('Actual data points', 'Predicted data points', 'Total message length'),
           loc='upper center', shadow=True)
plt.xlabel('n')
plt.ylabel('y(n)')
plt.title('Regression model')
plt.show()

#Prediction
n=m-2
x1_test = [x1[n-2], x1[n-1]]
x2_test = [x2[n-1]]

y = []
for i in range(30):
    y.append(theta1 * np.power(x1_test[i]/scale, order1) + theta2 * np.power(x2_test[i]/scale, order2))
    if i > 0:
        x1_test.append(y[i-1]*scale)
    x2_test.append(y[i]*scale)
y1 = []
for i in y:
    y1.append(i*scale)
print('Prediction of next 30 time units = ', y1)

#Accuracy
avg_err = 0
testfile = input('Test File name = ')      #load the test data points xlsx file here
data1 = xlrd.open_workbook(testfile)
sheet1 = data1.sheet_by_index(0)
test_data = sheet1.col_values(0)

for k in range(30):
        avg_err = avg_err + ((abs(y1[k] - test_data[k]))/(test_data[k]))
avg_err = 100 * avg_err/len(test_data)
print('Error percentage = ', avg_err, '%')
