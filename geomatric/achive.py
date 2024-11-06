"""
双塔结构

class GAT_GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, training=True):
        super().__init__()
        self.training = training
        self.conv1_t = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=args.heads,
                               dropout=args.drop)
        self.conv2_t = GATConv(in_channels=args.heads * hidden_channels, out_channels=dataset.num_classes, heads=1,
                               dropout=args.drop)

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        self.lin1 = nn.Linear(dataset.num_classes * 2, hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, dataset.num_classes)

        self.weight_t = torch.tensor(0.611, requires_grad=True)
        self.weight_c = torch.tensor(0.212, requires_grad=True)

    def forward(self, x, edge_index):
        x_gat = F.dropout(x, p=args.drop, training=self.training)
        x_gat = self.conv1_t(x_gat, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = F.dropout(x_gat, p=args.drop, training=self.training)
        x_gat = self.conv2_t(x_gat, edge_index)

        x_gcn = self.conv1(x, edge_index)
        x_gcn = x_gcn.relu()
        x_gcn = F.dropout(x_gcn, p=args.drop, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)

        x_cat = torch.cat([x_gcn * self.weight_c, x_gat * self.weight_t], dim=1)
        y = self.lin1(x_cat)
        y = self.lin2(y)
        y = self.lin3(y)

        return y


"""

"""

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.lin1 = nn.Linear(dataset.num_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=args.drop, training=self.training)
        y = self.lin2(x1)  # acc = 0.55
        # y = self.lin2(x1+x)
        return y



def train_mlp():
    data = load_data()
    model = MLP(hidden_channels=args.hidden, dataset=data)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # Define optimizer.

    def train():
        model.train()
        model.to(device=_device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def eva():
        model.eval()
        out = model(data.x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    max_acc = -1
    records = []
    for cur_epoch in range(1, 100):
        loss = train()
        test_acc = eva()
        if (max_acc < test_acc):
            max_acc = test_acc
            if (max_acc > args.min_acc):
                save_model(model=model, **vars(args))
        if (cur_epoch % 30 == 0):
            print_loss(epoch=cur_epoch, loss=loss.item(), test_acc=test_acc)
            records.append({'epoch': cur_epoch, 'loss': loss.item(), 'test_acc': test_acc})
    args.max_acc = max_acc
    save_json(records=records, **vars(args))
    print(max_acc)

"""

"""
# python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# pip uninstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv 
libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
- https://stackoverflow.com/questions/77664847/although-installed-pyg-lib-successfully-getting-error-while-importing
- Reference: What's the difference between "pip install" and "python -m pip install"?



TransformerConv
AGNNConv
FastRGCNConv
GCN

10-29 
在Core数据集上，使用Conv1d做残差网络的效果并不好 
在Core数据集上，使用隐藏层直接相加，作为残差，的效果比不上，直接输入上一层的，不做残差
在随机生成的数据集上， 使用隐藏层直接相加，作为残差，的效果优胜于，直接输入上一层的，不做残差

10-30 
使用AGN+GCN双塔结构
使用GCN+MLP双塔结构
使用AGN+MLP双塔结构 the experiment is not good 


10-31 图结点的分类问题，有最佳的模型
- https://paperswithcode.com/paper/optimization-of-graph-neural-networks-with
- https://github.com/russellizadi/ssp
- https://arxiv.org/pdf/2008.09624

11-1
- 通过实验，发现rest-net是有效的
- 通过实验，发现gat+gcn是有效的
不足之处：
1、模型不能过于简单
2、工作量

11-2 为此，准备2部分工作
- 工作量：加数据集、加任务、
- 模型：要设计时，复杂一些


11-6
    name=ResGCN_ds=MUTAG_ds_split=public_max_acc=0.89474_
    name=GCN_ds=MUTAG_ds_split=public_max_acc=0.86842_
    ds=naphthalene发生了一个异常: expected scalar type Long but found Float,
    ds=QM9发生了一个异常: The size of tensor a (64) must match the size of tensor b (19) at non-singleton dimension 1,
    ds=salicylic_acid发生了一个异常: expected scalar type Long but found Float,
    toluene dataset need more memory
    Mutagenicity so long time
    COIL-RAG low precision
    'MixHopConv', 'DirGNNConv', 'AntiSymmetricConv' 后面两个都需要有向图，前一个的输出形状会和power数组绑定
    ['GCNConv', 'GATConv', 'TransformerConv', 'MixHopConv', 'DirGNNConv', 'AntiSymmetricConv']
    ['BlockGNN' ,'ResBlockGnn','ResNodeBlockGnn','ResGraphBlockGnn','GraphBlockGnn']
"""


