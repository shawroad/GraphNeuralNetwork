cora数据集介绍

节点个数: 2708
边: 5429
特征: 1433
标签: 7
ind.cora.x: 训练实例的特征向量(针对节点的),size=(140,1433)
ind.cora.tx: 测试实例的特征向量(针对节点的),ize=(1000, 1433)
ind.cora.allx: 有标签和无标签训练实例的特征向量(针对节点的),size=(1708, 1433),相当于是所有的节点除去测试节点剩余的节点

ind.cora.y: 训练实例的标签(用one_hot表示)，和上面的ind.cora.x对应，size=(140, 7)
ind.cora.ty: 测试实例的标签(用one_hot表示)，和上面的ind.cora.tx对应，size=(1000, 7)
ind.cora.ally: 和上面ind.cora.allx对应，也是用one_hot表示标签，size=(1708, 7)

ind.cora.graph: 图数据，字典的格式[{'sour_node1': [dst_node1, dst_node2]}, {'sour_node1': [dst_node1, dst_node2]}...]
                上述数据表示sour_node1和dst_node1以及dst_node2都有一条边连接。
ind.cora.test.index: 测试实例的id, 2175行



