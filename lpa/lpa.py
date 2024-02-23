import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities as lpa
# 空手道俱乐部
G = nx.karate_club_graph()
com = list(lpa(G))
print(com)
print('社区数量',len(com))

# 下面是画图
pos = nx.spring_layout(G)
# 节点的布局为spring型
NodeId = list(G.nodes())
node_size = [G.degree(i)**1.2*90 for i in NodeId] # 节点大小
plt.figure(figsize = (8,6))
# 设置图片大小
nx.draw(G,pos, with_labels=True, node_size =node_size, node_color='w', node_shape = '.' )
'''node_size表示节点大小node_color表示节点颜色with_labels=True表示节点是否带标签'''
color_list = ['pink','orange','r','g','b','y','m','gray','black','c','brown']
for i in range(len(com)):
    nx.draw_networkx_nodes(G, pos, nodelist=com[i], node_color = color_list[i+2], label=True)
    plt.show()