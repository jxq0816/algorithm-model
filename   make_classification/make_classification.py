### 创建模型
def create_model():
    # 生成数据
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10000,  # 样本个数
                               n_features=25,  # 特征个数
                               n_informative=3,  # 有效特征个数
                               n_redundant=2,  # 冗余特征个数（有效特征的随机组合）
                               n_repeated=0,  # 重复特征个数（有效特征和冗余特征的随机组合）
                               n_classes=3,  # 样本类别
                               n_clusters_per_class=1,  # 簇的个数
                               random_state=0)

    print("原始特征维度", X.shape)

    # 读取数据
    print("读取数据")
    # import pandas as pd
    # data = pd.read_csv(datapath)

    # 数据划分
    print("数据划分")
    from sklearn.model_selection import train_test_split
    global x_train, x_valid, x_test, y_train, y_valid, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=33, test_size=0.25)

    # 创建模型
    print("创建模型")
    from sklearn.linear_model import LogisticRegression
    global model
    model = LogisticRegression(penalty='l2').fit(x_train, y_train)


### 保存模型
def save_model():
    print("保存模型")
    import joblib
    joblib.dump(model, 'model.pkl')


### 模型验证
def validate_model():
    print("模型验证")
    print(model.score(x_valid, y_valid))


### 模型预测
def predict_model():
    print("模型预测")
    global pred
    pred = model.predict_proba(x_test)
    print(pred)


if __name__ == "__main__":
    create_model()
    save_model()
    validate_model()
    predict_model()