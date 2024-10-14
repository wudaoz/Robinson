
from typing import Any

import numpy as np
import torch
from gnn_models import GCN, AMGapsGNN
from train_func import test, train


class Trainer_General:
    def __init__(
        self,
        rank: int,
        adj: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        idx_train: torch.Tensor,
        idx_test: torch.Tensor,
        args_hidden: int,
        class_num: int,
        device: torch.device,
        args: Any,
        attack_index: torch.Tensor,
    ):

        torch.manual_seed(rank)


        self.model = GCN(
            nfeat=features.shape[1],
            nhid=args_hidden,
            nclass=class_num,
            dropout=0.5,
            NumLayers=args.num_layers,
        ).to(device)

        self.rank = rank  # rank = client ID

        self.device = device

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.learning_rate, weight_decay=5e-4
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_losses: list = []
        self.train_accs: list = []
        self.train_pres: list = []

        self.test_losses: list = []
        self.test_accs: list = []
        self.test_pres: list = []

        self.adj = adj.to(device)
        self.labels = labels.to(device)
        self.features = features.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_test = idx_test.to(device)

        self.local_step = args.local_step
        self.attack_index = attack_index
        
    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        # load global parameter from global server
        self.model.to("cpu")
        for (
            p,
            mp,
        ) in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def train(self, current_global_round: int) -> None:
        # clean cache
        torch.cuda.empty_cache()
        for iteration in range(self.local_step):
            self.model.train()

            loss_train, acc_train, pre_train = train(
                iteration,
                self.model,
                self.optimizer,
                self.features,
                self.adj,
                self.labels,
                self.idx_train,
            )
            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)
            self.train_pres.append(pre_train)

            print(f"Client {self.rank} | Global Round {current_global_round} | "
                        f"Local Step {iteration} | Train Acc: {acc_train:.4f} | Train Pre: {pre_train:.4f}")
                        
            loss_test, acc_test, pre_test = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)
            self.test_pres.append(pre_test)

            print(f"Client {self.rank} | Global Round {current_global_round} | "
              f"Local Step {iteration} | Test Acc: {acc_test:.4f} | Test Pre: {pre_test:.4f}")
            
            

    def local_test(self) -> list:
        local_test_loss, local_test_acc, local_test_pre = test(
            self.model, self.features, self.adj, self.labels, self.idx_test, self.attack_index
        )
        return [local_test_loss, local_test_acc, local_test_pre]

    def get_params(self) -> tuple:
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())

    def get_all_loss_accuray(self) -> list:
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.train_pres),
            np.array(self.test_losses),
            np.array(self.test_accs),
            np.array(self.test_pres),
        ]

    def get_rank(self) -> int:
        return self.rank
    
    def get_accuracy(self) -> float: 

        return self.test_accs[-1] if self.test_accs else 0.0