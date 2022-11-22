import pandas as pd
import numpy as np
import math
def generate(b0,b1,b2,b3,b4,num=25000):
    x = pd.DataFrame()
    x[0] = np.random.randint(0,1,num)+1 # 全1
    x[1] = np.random.randint(low=12,high=60,size=num)
    x[2] = np.random.random(size = (num,1))*60 + 40
    x[3] = np.random.binomial(1,.45,num)
    x[4] = np.random.binomial(1,.55,num)
    x[5] = np.random.normal(0,1,num)
    beta = np.matrix([b0,b1,b2,b3,b4]).T
    X = np.matrix([x[0],x[1],x[2],x[3],x[4]]).T
    x[6] = X @ beta + np.matrix(x[5]).reshape((num,1))
    y = []
    for i in range(num):
        y.append(int(np.around(1/(1+np.exp(-x[6][i])))))
    x[7]=y
    res = x[[1,2,3,4,7]]
    res.columns = ['x1', 'x2', 'x3', 'x4', 'y', ]
    return res

data = generate(-.9,-0.2,0.05,0.5,0.3,500000)#5个系数（包括截距项）+生成的样本数量
print(data)
data.to_csv("fakedata_generated.csv",index=False)