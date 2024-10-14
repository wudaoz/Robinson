from typing import Any

import ray
import torch
from gnn_models import GCN, GCN_arxiv, SAGE_products
from trainer_class import Trainer_General


class Server:
    def __init__(
        self,
        feature_dim: int,
        args_hidden: int,
        class_num: int,
        device: torch.device,
        trainers: list[Trainer_General],
        args: Any,
    ) -> None:
        # server model on cpu
        if args.dataset == "ogbn-arxiv":
            self.model = GCN_arxiv(
                nfeat=feature_dim,
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            )
        elif args.dataset == "ogbn-products":
            self.model = SAGE_products(
                nfeat=feature_dim,
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            )
        else:  # CORA, CITESEER, PUBMED, REDDIT
            self.model = GCN(
                nfeat=feature_dim,
                nhid=args_hidden,
                nclass=class_num,
                dropout=0.5,
                NumLayers=args.num_layers,
            )

        self.trainers = trainers
        self.num_of_trainers = len(trainers)
        self.broadcast_params(-1)
        self.top_k = args.top_k  # 保存 top_k 参数

    @torch.no_grad()
    def zero_params(self) -> None:
        for p in self.model.parameters():
            p.zero_()

    @torch.no_grad()
    # def train(self, current_global_epoch: int) -> None:
    #     for trainer in self.trainers:
    #         trainer.train.remote(current_global_epoch)
    #     params = [trainer.get_params.remote() for trainer in self.trainers]
    #     self.zero_params()

    #     while True:
    #         ready, left = ray.wait(params, num_returns=1, timeout=None)
    #         if ready:
    #             for t in ready:
    #                 for p, mp in zip(ray.get(t), self.model.parameters()):
    #                     mp.data += p.cpu()
    #         params = left
    #         if not params:
    #             break

    #     for p in self.model.parameters():
    #         p /= self.num_of_trainers
    #     self.broadcast_params(current_global_epoch)
    
    def train(self, current_global_epoch: int) -> None:  #选topk的新train函数
        # 触发所有客户端的训练
        for trainer in self.trainers:
            trainer.train.remote(current_global_epoch)

        # 获取所有客户端的参数和性能
        params_and_accs = [
            (trainer.get_params.remote(), trainer.get_accuracy.remote())
            for trainer in self.trainers
        ]
        
        # 获取并排序客户端的准确率
        params_and_accs = [(p, ray.get(a)) for p, a in params_and_accs]
        params_and_accs.sort(key=lambda x: x[1], reverse=True)

        # 只保留 top_k 个客户端的参数
        selected_params = [p[0] for p in params_and_accs[:self.top_k]]

        self.zero_params()

        # 融合 top_k 客户端的参数
        while selected_params:
            ready, selected_params = ray.wait(selected_params, num_returns=1, timeout=None)
            for p, mp in zip(ray.get(ready[0]), self.model.parameters()):
                mp.data += p.cpu()

        # 平均融合的参数
        for p in self.model.parameters():
            p /= self.top_k

        self.broadcast_params(current_global_epoch)    

    def broadcast_params(self, current_global_epoch: int) -> None:
        for trainer in self.trainers:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )  # run in submit order