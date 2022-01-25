import pandas as pd
import os
import category_encoders
from category_encoders import TargetEncoder

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

# 训练集处理
print("训练集处理")
train_process = train.set_index(['PassengerId'])
train_process = train_process.drop(['Cabin'],axis=1)

# 获取名称中的称谓类别
train_process['Called'] = train_process['Name'].str.findall('Miss|Mr|Ms').str[0].to_frame()
# 得到名称长度
train_process['Name_length'] = train_process['Name'].apply(lambda x:len(x))
# 得到FirstName
train_process['First_name'] = train_process['Name'].str.split(',').str[0]
train_process = train_process.drop(['Name'],axis=1)

# 测试集处理
print("测试集处理")
test_process = test.set_index(['PassengerId'])
test_process = test_process.drop(['Cabin'],axis=1)

test_process['Called'] = test_process['Name'].str.findall('Miss|Mr|Ms').str[0].to_frame()
test_process['Name_length'] = test_process['Name'].apply(lambda x:len(x))
test_process['First_name'] = test_process['Name'].str.split(',').str[0]
test_process = test_process.drop(['Name'],axis=1)

print(train_process.select_dtypes('O'))
print(train_process.select_dtypes('number'))
print(test_process.select_dtypes('O'))
print(test_process.select_dtypes('number'))

# 特征编码(Target encoding)过程 
# target encoding其实就是将分类特征替换为对应目标值的后验概率
X_train = train_process.iloc[:,1:] # loc由索引搜索 iloc由数字
y_train = train_process.iloc[:,0]
X_test = test_process

tar_encoder1 = TargetEncoder(cols=['Sex','Ticket','Embarked','Called','Name_length','First_name'],
                             handle_missing='value',
                             handle_unknown='value')
tar_encoder1.fit(X_train,y_train)

X_train_encode = tar_encoder1.transform(X_train)
X_test_encode = tar_encoder1.transform(X_test)

print("编码后：")
print(X_train_encode)
print(X_test_encode)

os.system("pause")
