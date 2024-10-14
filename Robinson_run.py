import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
# import yaml
# from torch.utils.tensorboard import SummaryWriter

print(os.getcwd())
print(os.listdir())
print(os.listdir(".."))



ray.init()

from data_process import load_data
from gnn_models import GCN, AMGapsGNN
from server_class import Server
from trainer_class import Trainer_General
from utils import (
    get_in_comm_indexes,
    get_in_comm_indexes_BDS_GCN,
    increment_dir,
    label_dirichlet_partition,
    parition_non_iid,
    setdiff1d,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="theia", type=str)

    parser.add_argument("-f", "--fedtype", default="Robinson", type=str)

    parser.add_argument("-c", "--global_rounds", default=1, type=int)
    parser.add_argument("-i", "--local_step", default=50, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1, type=float)

    parser.add_argument("-n", "--n_trainer", default=5, type=int)
    parser.add_argument("-nl", "--num_layers", default=2, type=int)
    parser.add_argument("-nhop", "--num_hops", default=2, type=int)
    parser.add_argument("-g", "--gpu", action="store_true")  # if -g, use gpu
    parser.add_argument("-iid_b", "--iid_beta", default=10, type=float)

    parser.add_argument("-l", "--logdir", default="./runs", type=str)

    parser.add_argument("-r", "--repeat_time", default=1, type=int)
    parser.add_argument("--top_k", type=int, default=2, help="Number of top clients to select for aggregation")


    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    # load data to cpu
    
    features, adj, labels, idx_train, idx_test, original_attack = load_data(args.dataset)
    # class_num = labels.max().item() + 1
    class_num = int(labels.max().item() + 1) #增加int()，否则报错
    print("class_num is:", class_num)

        

    if args.dataset in ["cadets", "theia", "trace", "streanspot", "unicorn"]:
        args_hidden = 32
    else:
        args_hidden = 128

    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)

    # specifying a target GPU
    if args.gpu:
        device = torch.device("cuda")
        # running on a local machine with multiple gpu
        if args.dataset == "ogbn-products":
            edge_index = edge_index.to("cuda:7")
        else:
            edge_index = edge_index.to("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cpu":
        num_cpus = 0.1
        num_gpus = 0.0

    else:
        num_cpus = 10
        num_gpus = 1.0

    # repeat experiments
    average_final_test_loss_repeats = []
    average_final_test_accuracy_repeats = []
    average_final_test_precision_repeats = []

    for repeat in range(args.repeat_time):
        # load data to cpu

        # beta = 0.0001 extremely Non-IID, beta = 10000, IID
        split_data_indexes = label_dirichlet_partition(
            labels, len(labels), class_num, args.n_trainer, beta=args.iid_beta
        )

        for i in range(args.n_trainer):
            split_data_indexes[i] = np.array(split_data_indexes[i])
            split_data_indexes[i].sort()
            split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        (
            communicate_indexes,
            in_com_train_data_indexes,
            in_com_test_data_indexes,
            edge_indexes_clients,
        ) = get_in_comm_indexes(
            edge_index,
            split_data_indexes,
            args.n_trainer,
            args.num_hops,
            idx_train,
            idx_test,
        )

        # determine the resources for each trainer
        @ray.remote(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            scheduling_strategy="SPREAD",
        )
        class Trainer(Trainer_General):
            def __init__(self, *args: Any, **kwds: Any):
                super().__init__(*args, **kwds)


        trainers = [
            Trainer.remote(
                i,
                edge_indexes_clients[i],
                labels[communicate_indexes[i]],
                features[communicate_indexes[i]],
                in_com_train_data_indexes[i],
                in_com_test_data_indexes[i],
                args_hidden,
                class_num,
                device,
                args,
                attack_index=original_attack,
            )
            for i in range(args.n_trainer)
        ]

        torch.cuda.empty_cache()
        server = Server(
            features.shape[1], args_hidden, class_num, device, trainers, args
        )
        print("global_rounds", args.global_rounds)
        for i in range(args.global_rounds):
            server.train(i)

        results = [trainer.get_all_loss_accuray.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])


        train_data_weights = [len(i) for i in in_com_train_data_indexes]
        test_data_weights = [len(i) for i in in_com_test_data_indexes]

        average_train_loss = np.average(
            [row[0] for row in results], weights=train_data_weights, axis=0
        )
        average_train_accuracy = np.average(
            [row[1] for row in results], weights=train_data_weights, axis=0
        )
        average_train_preision = np.average(
            [row[2] for row in results], weights=train_data_weights, axis=0
        )
        average_test_loss = np.average(
            [row[3] for row in results], weights=test_data_weights, axis=0
        )
        average_test_accuracy = np.average(
            [row[4] for row in results], weights=test_data_weights, axis=0
        )
        average_test_preision = np.average(
            [row[5] for row in results], weights=test_data_weights, axis=0
        )



        results = [trainer.local_test.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])

        average_final_test_loss = np.average(
            [row[0] for row in results], weights=test_data_weights, axis=0
        )
        average_final_test_accuracy = np.average(
            [row[1] for row in results], weights=test_data_weights, axis=0
        )
        average_final_test_precision = np.average(
            [row[2] for row in results], weights=test_data_weights, axis=0
        )

        print(average_final_test_loss, average_final_test_accuracy, average_final_test_precision)

        # sleep(5)  # wait for print message from remote workers
        filename = (
            args.dataset
            + "_"
            + args.fedtype
            + "_"
            + str(args.num_layers)
            + "_layer_"
            + str(args.num_hops)
            + "_hop_iid_beta_"
            + str(args.iid_beta)
            + "_n_trainer_"
            + str(args.n_trainer)
            + "_local_step_"
            + str(args.local_step)
            + ".txt"
        )
        with open(filename, "a+") as a:
            a.write(f"{average_final_test_loss} {average_final_test_accuracy} {average_final_test_precision}\n")
            average_final_test_loss_repeats.append(average_final_test_loss)
            average_final_test_accuracy_repeats.append(average_final_test_accuracy)
            average_final_test_precision_repeats.append(average_final_test_precision)

    # finish experiments

    with open(
        f"cadets_all_acc.txt",
        "a+",
    ) as a:
        a.write(
            f"average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}\n"
        )

    print(
        f"average_testing_loss {np.average(average_final_test_loss_repeats)} std {np.std(average_final_test_loss_repeats)}"
    )
    print(
        f"average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}"
    )
    print(
        f"average_testing_precision {np.average(average_final_test_precision_repeats)} std {np.std(average_final_test_precision_repeats)}"
    )
ray.shutdown()
