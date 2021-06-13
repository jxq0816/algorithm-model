import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import DBSCAN
import random

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_pic(label_y):
    df_ddbb=df_splide_sam[df_splide_sam['label']==label_y]
    plt.figure(figsize=(10,10))
    if len(df_ddbb)>60000:
        df_ddbb=df_ddbb.sample(n=60000)
    print('count {}'.format(len(df_ddbb)))
    df_ddbb=df_ddbb.reset_index(drop=True)
    for i in range(len(df_ddbb)):
    #for i in range(100):
        plt.arrow(df_ddbb.loc[i,'x_1'],df_ddbb.loc[i,'y_1'],df_ddbb.loc[i,'x_dt'],df_ddbb.loc[i,'y_dt'],
                 length_includes_head=True,# 增加的长度包含箭头部分
                 head_width=10,
                 head_length=15,
                 fc='black',
                 ec='k',
                 alpha=0.01)
    #plt.axis('equal')
    plt.xlim((-50,1300))
    plt.ylim((-200,2000))
    plt.xlabel('lebel:{}'.format(label_y))
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')#将X坐标轴移到上面
    ax.invert_yaxis()#反转Y坐标轴
    plt.show()

def get_angle(delta_x,delta_y):
    length=np.sqrt(delta_x**2+delta_y**2)
    cos=delta_x/length
    angle=np.arccos(cos)*180/np.pi
    is_hpi=int(delta_y<0)
    ang=is_hpi*360+(-2*is_hpi+1)*angle
    if ang>180:
        ang=ang-180
    return round(ang,4)

df=pd.read_csv('plan4.csv',header=None,sep='\t')
df.head()
df.columns=['wm_type','wm_name_type','wm_name','wm','uid','anti','x','y','seid','method','x_1','y_1','x_2','y_2']
df=df.drop(['wm_name_type','anti','seid'],axis=1)
df.head()
df.dropna(axis=0, how='any', inplace=True)
df_splide=df[df.method==0]
del df_splide['method']