from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
max_min_scaler = lambda x : (x-np.mean(x))/np.std(x)

data = pd. read_csv("fakedata_generated.csv")
data['x1']=data[['x1']].apply(max_min_scaler)
data['x2']=data[['x2']].apply(max_min_scaler)
data['x0'] = np.random.randint(0,1,len(data['x1']))+1
data = data.loc[data.index<10000]
X =data[['x1','x2','x3','x4']]
Y = data['y']
model = LogisticRegression(solver='liblinear',max_iter=10000)
re = model.fit(X,Y)
print(re.coef_)