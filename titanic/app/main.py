import pandas as pd
import os

train = pd.read_csv("../dataset/train.csv") # header=None, names=range(2,6)
test = pd.read_csv("../dataset/test.csv")
print(train)

row_train = train.shape[0]
col_train = train.shape[1]
print(row_train)
print(col_train)

# 这后半边lambda写法，秀
print((train.isna().sum()/train.shape[0]).apply(lambda x:format(x,'.2%')))

# 数据集中的类别型变量和数值型变量
print(train.select_dtypes('O'))
print(train.select_dtypes('number'))

os.system("pause")
