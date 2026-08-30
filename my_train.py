import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from data_preprocess import preprocess_data
import joblib

data=pd.read_csv('train.csv')
# print(data.head(10)) # 读取前10行
# print(data.info()) # 返回表格信息
# print(data.isnull().sum()) # 每列缺失值求和
# 缺失：age(部分缺失，平均数填充),cabin(客舱号，大量缺失，删掉这一列),embarked(登船港口，少量缺失，用众数填充)

data=preprocess_data(data)
# 训练

y=data['Survived']
x=data.drop(columns='Survived') # 把答案去掉
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print(accuracy)

# 测试
test_data=pd.read_csv("test.csv")
test_ids=test_data['PassengerId']
test_data=preprocess_data(test_data)

prediction = model.predict(test_data)
# print(prediction)
submission=pd.DataFrame(
    {
        'PassengerId':test_ids,
        'Survived':prediction
    }
)
submission.to_csv('my_submission.csv',index=False)


# 保存模型
joblib.dump(model,'titanic_model.pkl')