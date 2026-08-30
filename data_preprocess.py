import pandas as pd
import re
def preprocess_data(df):
    data=df.copy()
    # 数据清洗
    # 1.缺失值处理
    mean_age = data['Age'].mean()
    data['Age'] = data['Age'].fillna(mean_age)

    data = data.drop(columns=['Cabin'])
    x = data['Embarked'].mode()[0]  # 列表（索引+值）
    # print(x)
    data['Embarked'] = data['Embarked'].fillna(x)
    # print(data.isnull().sum())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data_name=data['Name']
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket'])



    # 3.特征工程
    # 家庭规模
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    # 是否独自一人
    data['IsAlone']=(data['FamilySize']==1).astype(int)
    # 从名字中提取称呼
    data['Title']=data_name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # print(pd.crosstab(data['Title'],data['Sex']))
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs',
        'Master': 'Master', 'Dr': 'Rare', 'Rev': 'Rare',
        'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss',
        'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare',
        'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data=pd.get_dummies(data,columns=['Title'])


    # 2.类别数据编码(sex,embarked,passengerId,name,ticket)
    # 用one hot encoding
    # data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    # print(data['Sex'].head())
    # data['Embarked'] = data['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
    data = pd.get_dummies(data, columns=['Sex'])
    data = pd.get_dummies(data, columns=['Embarked'])

    return data

data=pd.read_csv('train.csv')
data=preprocess_data(data)