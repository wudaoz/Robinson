from typing import Any

import ray
import torch
from gnn_models import GCN, AMGapsGNN
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
        self.top_k = args.top_k  

    @torch.no_grad()
    def zero_params(self) -> None:
        for p in self.model.parameters():
            p.zero_()

    @torch.no_grad()

    
    def train(self, current_global_epoch: int) -> None:

        for trainer in self.trainers:
            trainer.train.remote(current_global_epoch)


        params_and_accs = [
            (trainer.get_params.remote(), trainer.get_accuracy.remote())
            for trainer in self.trainers
        ]
        

        params_and_accs = [(p, ray.get(a)) for p, a in params_and_accs]
        params_and_accs.sort(key=lambda x: x[1], reverse=True)


        selected_params = [p[0] for p in params_and_accs[:self.top_k]]

        self.zero_params()

        while selected_params:
            ready, selected_params = ray.wait(selected_params, num_returns=1, timeout=None)
            for p, mp in zip(ray.get(ready[0]), self.model.parameters()):
                mp.data += p.cpu()


        for p in self.model.parameters():
            p /= self.top_k

        self.broadcast_params(current_global_epoch)    

    def broadcast_params(self, current_global_epoch: int) -> None:
        for trainer in self.trainers:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )  # run in submit order