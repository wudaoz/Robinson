import torch
import torch.nn.functional as F
import torch_sparse

def get_two_hop_neighbors(adj: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
    # 如果 node_idx 是 set 类型，先转换为 Tensor
    if isinstance(node_idx, set):
        node_idx = torch.tensor(list(node_idx))
    
    row, col, _ = adj.coo() 
    one_hop_mask = torch.isin(row, node_idx)
    one_hop_neighbors = col[one_hop_mask]
    
    two_hop_mask = torch.isin(row, one_hop_neighbors)
    two_hop_neighbors = col[two_hop_mask]

    # three_hop_mask = torch.isin(row, two_hop_neighbors)
    # three_hop_neighbors = col[three_hop_mask]
    
    return two_hop_neighbors


# def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
#     """
#     This function returns the accuracy of the output with respect to the ground truth given

#     Arguments:
#     output: (torch.Tensor) - the output labels predicted by the model

#     labels: (torch.Tensor) - ground truth labels

#     Returns:
#     The accuracy of the model (float)
#     """
#     # print("output shape are:", output.shape)
#     # print(output.max(1)[1].type_as(labels))
#     # print("labels shape are:", labels.shape)

#     # preds = output.max(1)[1].type_as(labels)
#     # correct = preds.eq(labels).double()
#     # correct = correct.sum()
#     # return correct / len(labels)

#     # 获取模型的预测类别
#     preds = output.max(1)[1].type_as(labels)
#     print("preds shape are:", preds.shape)
    
    
#     # 计算TP, TN, FP, FN
#     TP = ((preds == 1) & (labels == 1)).sum().item()
#     TN = ((preds == 0) & (labels == 0)).sum().item()
#     FP = ((preds == 1) & (labels == 0)).sum().item()
#     FN = ((preds == 0) & (labels == 1)).sum().item()

#     print("TP, TN, FP, FN are:", TP, TN, FP, FN)

#     if (TP + TN + FP + FN) == 0:
#         acc = 0
#     else:  
#         acc = (TP + TN) / (TP + TN + FP + FN)

#     if (TP + FP) == 0:
#         pre = 0
#     else:
#         pre = TP / (TP + FP)
#     return acc, pre

def accuracy_train(output: torch.Tensor, labels: torch.Tensor, adj: torch.Tensor, index: torch.Tensor) -> float:   #两跳版本的新Accuracy函数
    """
    This function returns the accuracy of the output with respect to the ground truth given

    Arguments:
    output: (torch.Tensor) - the output labels predicted by the model

    labels: (torch.Tensor) - ground truth labels

    Returns:
    The accuracy of the model (float)
    """
    print("output shape are:", output.shape)
    print(output.max(1)[1].type_as(labels))
    print("labels shape are:", labels.shape)

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item() 
    if len(labels) == 0:
        print("Error: No labels available for computing accuracy!")
    return correct / len(labels)

    # # 获取模型的预测类别
    # preds = output.max(1)[1].type_as(labels)
    # print("preds shape are:", preds.shape)
    # two_hop_neighbors = get_two_hop_neighbors(adj, index)
    
    # # 计算TP, TN, FP, FN
    # TP = ((preds == 1) & (labels == 1)).sum().item()
    # TN = ((preds == 0) & (labels == 0)).sum().item()
    # FP = ((preds == 1) & (labels == 0)).sum().item()
    # FN = ((preds == 0) & (labels == 1)).sum().item()

    # print("TP, TN, FP, FN are:", TP, TN, FP, FN)
    # # 计算 FPL
    # fp_mask = (preds == 1) & (labels == 0)
    # fp_indices = fp_mask.nonzero(as_tuple=False).squeeze(1)

    # # 找到假阳性节点中属于两跳邻居的部分
    # fp_in_two_hop = torch.isin(fp_indices, two_hop_neighbors).sum().item()

    # # 计算调整后的假阳性
    # FPL = FP - fp_in_two_hop
    # print("FPL is:", FPL)

    # # 计算 two_hop_tp - 真阳性节点集合的两跳邻居
    # tp_mask = (preds == 1) & (labels == 1)
    # tp_indices = tp_mask.nonzero(as_tuple=False).squeeze(1)
    # two_hop_tp = get_two_hop_neighbors(adj, tp_indices)

    # # 计算 FN 和 two_hop_tp 的交集
    # fn_mask = (preds == 0) & (labels == 1)
    # fn_indices = fn_mask.nonzero(as_tuple=False).squeeze(1)
    # fn_in_two_hop_tp = torch.isin(fn_indices, two_hop_tp)

    # # 更新 TPL
    # tpl_indices = torch.cat((tp_indices, fn_indices[fn_in_two_hop_tp])).unique()
    # TPL = len(tpl_indices)

    # print("TPL is:", TPL)

    # # 从 FN 中去掉被重新归类为 TP 的节点
    # FN_corrected = fn_indices[~fn_in_two_hop_tp]
    # FNL = len(FN_corrected)
    # print("FNL is:", FNL)

    # if (TPL + TN + FPL + FNL) == 0:
    #     acc = 0
    # else:  
    #     acc = (TPL + TN) / (TPL + TN + FPL + FNL)

    # if (TPL + FPL) == 0:
    #     pre = 0
    # else:
    #     pre = TPL / (TPL + FPL)
    
    # return acc, pre

def accuracy_test(output: torch.Tensor, labels: torch.Tensor, adj: torch.Tensor, index:torch.Tensor, attack_index:torch.Tensor) -> float:   #两跳版本的新Accuracy函数
    """
    This function returns the accuracy of the output with respect to the ground truth given

    Arguments:
    output: (torch.Tensor) - the output labels predicted by the model

    labels: (torch.Tensor) - ground truth labels

    Returns:
    The accuracy of the model (float)
    """
    # print("output shape are:", output.shape)
    # print(output.max(1)[1].type_as(labels))
    # print("labels shape are:", labels.shape)

    # preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    # return correct / len(labels)

    # ！！！注意以下是按照节点类型判断进行半监督的版本（cadets数据集）！！！
    # print("index number is:", len(index))
    # preds = output.max(1)[1].type_as(labels)
    # print("preds shape are:", preds.shape)
    # print("attack_index number is:", len(attack_index))
    # print("index number is:", len(index))
    # #返回attack_index和index的交集
    # GP = attack_index[torch.isin(attack_index, index)]
    # incorrect_mask = preds.ne(labels)
    # MP = index[incorrect_mask]
    #  # 转换为集合进行集合操作
    # GP_set = set(GP.tolist())
    # MP_set = set(MP.tolist())
    # attack = set(attack_index.tolist()) #试试水

    # print("GP_set number is:", len(GP_set)) #试试水，把GP_set改成attack
    # print("MP_set number is:", len(MP_set))

    # TP = MP_set.intersection(GP_set)
    # FP = MP_set - GP_set
    # FN = GP_set - MP_set
    # TN = set(index.tolist()) - (GP_set.union(MP_set))
    
    # two_hop_neighbors_gp = get_two_hop_neighbors(adj, GP)
    # two_hop_neighbors_tp = get_two_hop_neighbors(adj, TP)

    # print("two_hop_neighbors_gp number is:", len(two_hop_neighbors_gp))
    # print("two_hop_neighbors_tp number is:", len(two_hop_neighbors_tp))
    # print("TP, TN, FP, FN are:", len(TP), len(TN), len(FP), len(FN))

    # FPL = FP - set(two_hop_neighbors_gp.tolist())
    # TPL = TP.union(FN.intersection(set(two_hop_neighbors_tp.tolist())))
    # FNL = FN - set(two_hop_neighbors_tp.tolist())

    # if (len(TPL) + len(TN) + len(FPL) + len(FNL)) == 0:
    #     acc = 0
    # else:  
    #     acc = (len(TPL) + len(TN)) / (len(TPL) + len(TN) + len(FPL) + len(FNL))

    # if (len(TPL) + len(FPL)) == 0:
    #     pre = 0
    # else:
    #     pre = len(TPL) / (len(TPL) + len(FPL))
    # print("TPL, TNL, FPL, FNL are:", len(TPL), len(TN), len(FPL), len(FNL))
    
    # return acc, pre


    # !!!!!以下是有监督的版本！！！！有twohops，但是不通过GP、MP计算！！

    preds = output.max(1)[1].type_as(labels)
    print("preds shape are:", preds.shape)
    two_hop_neighbors = get_two_hop_neighbors(adj, index)
    
    # 计算TP, TN, FP, FN
    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    print("TP, TN, FP, FN are:", TP, TN, FP, FN)
    # 计算 FPL
    fp_mask = (preds == 1) & (labels == 0)
    fp_indices = fp_mask.nonzero(as_tuple=False).squeeze(1)

    # 找到假阳性节点中属于两跳邻居的部分
    fp_in_two_hop = torch.isin(fp_indices, two_hop_neighbors).sum().item()

    # 计算调整后的假阳性
    FPL = FP - fp_in_two_hop
    print("FPL is:", FPL)

    # 计算 two_hop_tp - 真阳性节点集合的两跳邻居
    tp_mask = (preds == 1) & (labels == 1)
    tp_indices = tp_mask.nonzero(as_tuple=False).squeeze(1)
    two_hop_tp = get_two_hop_neighbors(adj, tp_indices)

    # 计算 FN 和 two_hop_tp 的交集
    fn_mask = (preds == 0) & (labels == 1)
    fn_indices = fn_mask.nonzero(as_tuple=False).squeeze(1)
    fn_in_two_hop_tp = torch.isin(fn_indices, two_hop_tp)

    # 更新 TPL
    tpl_indices = torch.cat((tp_indices, fn_indices[fn_in_two_hop_tp])).unique()
    TPL = len(tpl_indices)

    print("TPL is:", TPL)

    # 从 FN 中去掉被重新归类为 TP 的节点
    FN_corrected = fn_indices[~fn_in_two_hop_tp]
    FNL = len(FN_corrected)
    print("FNL is:", FNL)

    if (TPL + TN + FPL + FNL) == 0:
        acc = 0
    else:  
        acc = (TPL + TN) / (TPL + TN + FPL + FNL)

    if (TPL + FPL) == 0:
        pre = 0
    else:
        pre = TPL / (TPL + FPL)
    

    #把acc存进txt文件中，标好轮
    return acc, pre


def test(
    model: torch.nn.Module,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    idx_test: torch.Tensor,
    attack_index: torch.Tensor,
) -> tuple:
    """
    This function tests the model and calculates the loss and accuracy

    Arguments:
    model: (torch.nn.Module) - Specific model passed
    features: (torch.Tensor) - Tensor representing the input features
    adj: (torch.Tensor) - Adjacency matrix
    labels: (torch.Tensor) - Contains the ground truth labels for the data.
    idx_test: (torch.Tensor) - Indices specifying the test data points

    Returns:
    The loss and accuracy of the model

    """
    model.eval()
    output = model(features, adj)
    pred_labels = torch.argmax(output, axis=1)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_test = loss_fn(output[idx_test], labels[idx_test])


    acc_test, pre_test = accuracy_test(output[idx_test], labels[idx_test], adj, idx_test, attack_index)

    return loss_test.item(), acc_test, pre_test  # , f1_test, auc_test


def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
) -> tuple:  # Centralized or new FL
    """
    This function trains the model and returns the loss and accuracy

    Arguments:
    model: (torch.nn.Module) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    adj: (torch_sparse.tensor.SparseTensor) - Adjacency matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data.
    idx_train: (torch.LongTensor) - Indices specifying the test data points
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_train = loss_fn(output[idx_train], labels[idx_train])


    acc_train = accuracy_train(output[idx_train], labels[idx_train], adj, idx_train)
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    pre_train=0
    return loss_train.item(), acc_train, pre_train


def Lhop_Block_matrix_train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    communicate_index: torch.Tensor,
    in_com_train_data_index: torch.Tensor,
) -> tuple:
    """
    Arguments:
    model: (model type) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    adj: (torch_sparse.tensor.SparseTensor) - Adjacency matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used
    communicate_index: (PyTorch tensor) - List of indices specifying which data points are used for communication
    in_com_train_data_index (PyTorch tensor): Q: Diff bet this and communicate index?

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()

    output = model(
        features[communicate_index], adj[communicate_index][:, communicate_index]
    )

    loss_train = F.nll_loss(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    acc_train, pre_train = accuracy(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_train.item(), acc_train, pre_train


def FedSage_train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    communicate_index: torch.Tensor,
    in_com_train_data_index: torch.Tensor,
) -> tuple:
    """
    This function is to train the FedSage model

    Arguments:
    model: (model type) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    adj: (torch_sparse.tensor.SparseTensor) - Adjacency matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used
    communicate_index: (PyTorch tensor) - List of indices specifying which data points are used for communication
    in_com_train_data_index (PyTorch tensor): Q: Diff bet this and communicate index?

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()
    # print(features.shape)

    output = model(features, adj[communicate_index][:, communicate_index])

    loss_train = F.nll_loss(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    acc_train, pre_train = accuracy(
        output[in_com_train_data_index],
        labels[communicate_index][in_com_train_data_index],
    )

    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss_train.item(), acc_train, pre_train
