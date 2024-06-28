import torch
import numpy as np

import argparse
import json
from os import path, environ
from time import time
import random

from utils import get_device
from framework.metadata import TaskMetadata
from framework.fully_neural import FullyNeuralModel
from framework.fully_neural_trainer import FullyNeuralTrainer, load_best_fully_neural_model
from framework.clustering import Clustering, ClusterTrainer
from framework.weak_labeller import WeakLabeller
from framework.cluster_labeller import ClusterLabeller, ClusterLabelOptimiser
from framework.symbolic_learner import SymbolicLearner
from framework.neurosymbolic_model import NeuroSymbolicModel


SUBTASKS = {
    "hitting_sets": [
        "5_4",
        "10_4",
        "10_5",
        "10_6",
    ],
    "sudoku": [
        "4_fixed",
        "4_random",
        "9_fixed",
    ],
    "follow_suit": [
        "4"
    ]
}

def get_args():
    parser = argparse.ArgumentParser("Run an experiment.")
    parser.add_argument("--task-name", type=str, required=True, help="Name of experiment to run.", choices=["hitting_sets", "sudoku", "follow_suit"])
    parser.add_argument("--subtask-name", type=str, required=True, help="Name of subtask to run.")
    parser.add_argument("--run-name", type=str, required=True, help="Name of run.")
    parser.add_argument("--repeats", type=int, required=True, help="Number of times to repeat experiment.")
    parser.add_argument("--seed", type=int, required=False, default=None, help="Set seed (random if not set).")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save experiment results to (json file).")
    parser.add_argument("--eval-split", type=str, required=False, default="test", help="Split to evaluate on.", choices=["train", "val", "test"])
    args = parser.parse_args()
    if args.subtask_name not in SUBTASKS[args.task_name]:
        raise ValueError(f"Subtask {args.subtask_name} not found for task {args.task_name}. Valid choices are {SUBTASKS[args.task_name]}.")
    return args


def set_seed(seed):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_random_seed():
    random_seed = random.randint(1, 10000)
    set_seed(random_seed)
    return random_seed

def get_additional_args(task_name, subtask_name):
    if task_name == "hitting_sets":
        return {}
    elif task_name == "sudoku":
        return {}
    elif task_name == "follow_suit":
        return {
            "cards_data_path": path.join("data", "CardsData", "standard")
        }
    else:
        raise ValueError(f"Unknown task {task_name}.")


def run_experiment(device, task_name, subtask_name, run_name, eval_split, seed, additional_args):
    experiment_path_base = path.join("experiments", task_name, str(subtask_name))
    data_path = path.join(experiment_path_base, "data.pkl")
    background_file = path.join(experiment_path_base, "background.lp")
    results_dir = path.join("results", task_name, str(subtask_name), run_name)
    hypothesis_file = path.join(results_dir, "hypothesis.lp")

    with open(path.join(experiment_path_base, "config.json"), "r") as f:
        config = json.load(f)

    metadata = TaskMetadata.from_json(path.join(experiment_path_base, "metadata.json"))

    task_module = __import__(f"experiments.{task_name}.task", fromlist=[""])
    latent_concept_datasets = {
        split: getattr(task_module, "get_latent_concept_datasets")(config, split, additional_args)
        for split in ["train", "val", "test"]
    }

    run_begin_time = time()

    perception_networks, reasoning_network = getattr(task_module, "get_networks")(config, metadata)
    fully_neural = FullyNeuralModel(metadata, perception_networks, reasoning_network).to(device)
    trainer = FullyNeuralTrainer(config, device, metadata, fully_neural, data_path, latent_concept_datasets,
                                 results_dir)

    fully_neural_train_begin = time()
    trainer.train(epochs=config["train"]["epochs"])
    fully_neural_train_end = time()
    fully_neural = load_best_fully_neural_model(fully_neural, results_dir).to(device)


    clustering = Clustering(metadata)
    cluster_trainer = ClusterTrainer(config, device, metadata, clustering, fully_neural.perception_networks, data_path, latent_concept_datasets)
    clustering_begin = time()
    cluster_trainer.train(results_dir)
    clustering_end = time()

    weak_labeller = WeakLabeller(config, device, metadata)
    cluster_labeller = ClusterLabeller(config, metadata)
    cluster_label_optimiser = ClusterLabelOptimiser(config, device, metadata, cluster_labeller, fully_neural,
                                                    clustering, weak_labeller, data_path, latent_concept_datasets)
    cluster_label_optimisation_begin = time()
    cluster_label_optimiser.optimise_cluster_labels(results_dir)
    cluster_label_optimisation_end = time()

    symbolic_learner = SymbolicLearner(config, device, metadata, fully_neural, clustering, cluster_labeller,
                                 experiment_path_base, latent_concept_datasets)
    symbolic_learning_begin = time()
    hypothesis = symbolic_learner.learn_hypothesis(results_dir)
    symbolic_learning_end = time()
    with open(path.join(results_dir, "hypothesis.lp"), "w") as f:
        f.write("\n".join(hypothesis))

    run_end_time = time()

    ns_model = NeuroSymbolicModel(config, device, metadata, fully_neural, clustering, cluster_labeller, background_file,
                                  hypothesis_file)

    fully_neural_results = ns_model.evaluate_fully_neural(data_path, eval_split, latent_concept_datasets)
    latent_results = ns_model.evaluate_latent(data_path, eval_split, latent_concept_datasets)
    neurosymbolic_results = ns_model.evaluate_neurosymbolic(data_path, eval_split, latent_concept_datasets)

    neural_training_time = fully_neural_train_end - fully_neural_train_begin
    clustering_time = clustering_end - clustering_begin
    cluster_labelling_time = cluster_label_optimisation_end - cluster_label_optimisation_begin
    symbolic_learning_time = symbolic_learning_end - symbolic_learning_begin
    run_time = run_end_time - run_begin_time

    print(f"Run name: {run_name}")
    print(f"Seed: {seed}")
    print(f"Fully neural accuracy: {fully_neural_results}")
    print(f"Latent accuracy: {latent_results}")
    print(f"Neurosymbolic accuracy: {neurosymbolic_results}")
    print(f"Neural training time: {neural_training_time:.2f}s")
    print(f"Clustering time: {clustering_time:.2f}s")
    print(f"Cluster labelling time: {cluster_labelling_time:.2f}s")
    print(f"Symbolic learning time: {symbolic_learning_time:.2f}s")
    print(f"Run time: {run_time:.2f}s")

    return {
        "run_name": run_name,
        "seed": seed,
        "fully_neural_acc": fully_neural_results,
        "latent_acc": latent_results,
        "neurosymbolic_acc": neurosymbolic_results,
        "neural_training_time": neural_training_time,
        "clustering_time": clustering_time,
        "cluster_labelling_time": cluster_labelling_time,
        "symbolic_learning_time": symbolic_learning_time,
        "run_time": run_time
    }

def main(args):
    device = get_device()
    if args.seed is not None:
        set_seed(args.seed)
        seed = args.seed
    else:
        seed = set_random_seed()

    results = list()
    if path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            results = json.load(f)

    for i in range(args.repeats):
        if args.repeats == 1:
            run_name = args.run_name
        else:
            run_name = f"{args.run_name}_{i+1}"
        print(f"Running repeat {i + 1} of {args.repeats}...")
        run_results = run_experiment(device=device,
                                     task_name=args.task_name,
                                     subtask_name=args.subtask_name,
                                     run_name=run_name,
                                     eval_split=args.eval_split,
                                     seed=seed,
                                     additional_args=get_additional_args(args.task_name, args.subtask_name))
        results.append(run_results)
        with open(args.output_path, "w") as f:
            f.write(json.dumps(results, indent=4))

if __name__ == "__main__":
    args = get_args()
    main(args)
