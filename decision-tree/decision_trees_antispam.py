from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time

from IPython.display import Image
from sklearn import tree
import pydotplus

def show(clf,features,y_types):
    """决策树的可视化"""
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features,
                                    class_names=y_types,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #Image(graph.create_png())  #jupyter里可以显示，pycharm显示不出
    graph.write_png(r'DT_show.png')

def main():
    star=time.time()
    # 原始样本数据
    features=['devicemodel','resolution']
    excelFile = 'devicemodel_resolution_map.csv'
    data=pd.DataFrame(pd.read_csv(excelFile))
    X_train=data[['devicemodel','resolution']]
    y_train = data[['tag']]
    # 数据预处理：不能处理文本
    le_x=preprocessing.LabelEncoder()
    le_x.fit(np.unique(X_train))
    X_train=X_train.apply(le_x.transform)
    print(X_train)
    le_y=preprocessing.LabelEncoder()
    le_y.fit(np.unique(y_train))
    y_train=y_train.apply(le_y.transform)
    # 调用sklearn.DT建立训练模型
    clf=DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    # 可视化
    show(clf,features,[str(k) for k in np.unique(y_train)])
    # 用训练得到模型进行预测
    X_new=pd.DataFrame([['xiaomi__xiaomi__mi 6', '1080x1921']])

    X=X_new.apply(le_x.transform)
    y_predict=clf.predict(X)
    # 结果输出
    X_show=[{features[i]:X_new.values[0][i]} for i in range(len(features))]
    print("{0}被分类为:{1}".format(X_show,le_y.inverse_transform(y_predict)))
    print("time:{:.4f}s".format(time.time()-star))

if __name__=="__main__":
    main()