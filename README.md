- 这是一个基于 Kaggle 经典赛事“泰坦尼克号幸存者预测”的机器学习入门项目。
- **本项目提供了一套完整的数据处理流水线，并使用随机森林（Random Forest）实现了基线预测模型。**
- 核心工程点 (Key Features)： 
	  1. **缺失值处理:** 针对 `Age`、`Fare` 采用均值填补，针对 `Embarked` 采用众数填补。 
	  2. **特征工程 :** 
	     * **Title 提取:** 利用正则表达式从 `Name` 中提取社会头衔（如 Mr, Miss, Master），并进行罕见头衔合并。
	    * **家庭规模:** 融合 `SibSp` 和 `Parch` 构建 `FamilySize` 及 `IsAlone` 特征，捕捉家庭存活共性。 
	  3. **独热编码 (One-Hot Encoding):** 利用 `pd.get_dummies` 对性别和登船港口等无序类别特征进行降维打击，避免引入虚假的数学大小关系
 - 运行结果 ：本地验证集准确率 :83% 
           **Kaggle 提交得分 :** 0.77315

   项目流程图：
   https://www.runoob.com/wp-content/uploads/2025/12/ml-titanic-survival-prediction-runoob-scaled.png
