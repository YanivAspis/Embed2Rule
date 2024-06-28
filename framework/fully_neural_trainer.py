import torch
from flash.core.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from os import path, makedirs
import math

from framework.dataset import TaskDataset
from framework.TSNE_plot_generator import TSNEPlotGenerator
from experiments.metrics import categorical_accuracy, binary_accuracy

class FullyNeuralTrainer:
    def __init__(self, config, device, metadata, fully_neural_model, data_path, latent_concept_datasets, results_dir):
        self._config = config
        self._device = device
        self._metadata = metadata
        self._results_dir = results_dir
        self.model = fully_neural_model
        self.model.to(device)
        self._setup_dataloaders(metadata, data_path, latent_concept_datasets)
        self._setup_loss_functions(metadata)
        self._setup_metrics(metadata)
        self._setup_visualiser(data_path, latent_concept_datasets)
        self._use_scheduler = "use_scheduler" in self._config["train"] and self._config["train"]["use_scheduler"]
        self._setup_optimisers()
        self._setup_schedulers()

    def _setup_dataloaders(self, metadata, data_path, latent_concept_datasets):
        self._train_loader = torch.utils.data.DataLoader(
            TaskDataset(metadata, data_path, latent_concept_datasets, "train"),
            batch_size = self._config["train"]["batch_size"],
            shuffle = True
        )
        self._val_loader = torch.utils.data.DataLoader(
            TaskDataset(metadata, data_path, latent_concept_datasets, "val"),
            batch_size=self._config["train"]["batch_size"],
            shuffle=False
        )

    def _setup_loss_functions(self, metadata):
        self._loss_funcs = dict()
        for target_name, target_data in metadata.targets.items():
            if target_data.multi_valued:
                self._loss_funcs[target_name] = torch.nn.BCELoss()
            else:
                self._loss_funcs[target_name] =torch.nn.CrossEntropyLoss()

    def _setup_metrics(self, metadata):
        self._metrics = dict()
        for target_name, target_data in metadata.targets.items():
            if target_data.multi_valued:
                self._metrics[target_name] = binary_accuracy
            else:
                self._metrics[target_name] = categorical_accuracy

    def _setup_logging(self):
        pass

    def _log(self, split, entry):
        log_line = "\t".join([str(key) + " = " + str(val) for key, val in entry.items()])
        print(log_line)

    def _setup_visualiser(self, data_path, latent_concept_datasets):
        self._visualiser = TSNEPlotGenerator(self._config, self._device, self._metadata, data_path, latent_concept_datasets, "val", self._results_dir)

    def _setup_weight_decay(self, model):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.Embedding)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # don't decay biases
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # decay whitelist modules' weights
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # don't decay weights of blacklist modules
                    no_decay.add(fpn)
                elif pn.endswith('proj_weight'):
                    # decay projection weights
                    decay.add(fpn)
                elif pn.endswith('pos_emb'):
                    # position embedding parameters are not decayed
                    no_decay.add(fpn)
                    decay.discard(fpn)
                elif pn.endswith("bn1.weight") or pn.endswith("bn2.weight"):
                    # Don't decay batch norm weights (specific case for loading in resnet)
                    no_decay.add(fpn)
                elif pn.endswith("downsample.1.weight"):
                    # Don't decay downsampling weights (specific case for loading in resnet)
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimiser object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups
  
    def _setup_optimisers(self):
        if self._use_scheduler:
            perception_optim_groups = self._setup_weight_decay(self.model.perception_networks)
            reasoning_optim_groups = self._setup_weight_decay(self.model.reasoning_network)

            self._perception_optimiser = torch.optim.AdamW(params=perception_optim_groups,
                                                            lr=self._config["train"]["learning_rate"],
                                                            betas=(0.9, 0.95))
            self._reasoning_optimiser = torch.optim.AdamW(params=reasoning_optim_groups,
                                                            lr=self._config["train"]["learning_rate"],
                                                            betas=(0.9, 0.95))
        else:
            self._perception_optimiser = torch.optim.Adam(params=self.model.perception_networks.parameters(),
                                                           lr=self._config["train"]["learning_rate"])
            self._reasoning_optimiser = torch.optim.Adam(params=self.model.reasoning_network.parameters(),
                                                          lr=self._config["train"]["learning_rate"])

    def _setup_schedulers(self):
        if self._use_scheduler:
            dataset_size = self._config["train"]['dataset_size']
            batch_size = self._config["train"]["batch_size"]
            epochs = self._config["train"]["epochs"]
            warmup_coefficient = self._config["train"]["warmup_coefficient"]
            minimum_learning_rate = self._config["train"]["minimum_learning_rate"]

            steps_per_epochs = dataset_size // batch_size
            total_steps = epochs * steps_per_epochs
            warmup_steps = math.floor(warmup_coefficient * total_steps)

            self._perception_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self._perception_optimiser,
                                                                        warmup_epochs=warmup_steps,
                                                                        max_epochs=total_steps,
                                                                        eta_min=minimum_learning_rate)
            self._reasoning_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self._reasoning_optimiser,
                                                                        warmup_epochs=warmup_steps,
                                                                        max_epochs=total_steps,
                                                                        eta_min=minimum_learning_rate)
    
    def _plot_visualisation(self, epoch):
        self._visualiser(self.model, epoch)

    def _tensor_list_to_device(self, tensor_list):
        return [tensor.to(self._device) for tensor in tensor_list]

    def train(self, epochs=None):
        self._best_loss = float("inf")
        epochs = self._config["epochs"] if epochs is None else epochs

        for epoch in range(epochs):
            if epoch == 0:
                self._plot_visualisation("init")
            self._train_epoch(epoch)
            epoch_loss = self.validate_epoch(epoch)
            self.save_model(epoch, epoch_loss)
            if self._config["visualisation"]["enabled"]:
                self._plot_visualisation(epoch)

    def _compute_loss(self, target_predictions, target_labels):
        return torch.sum(torch.stack([
            loss_func(preds, labels)
            for preds, labels, loss_func in zip(target_predictions, target_labels, self._loss_funcs.values())
        ]))

    @torch.no_grad()
    def _compute_metrics(self, target_predictions, target_labels):
        return {
            target_name: metric(preds, labels).item()
            for preds, labels, (target_name, metric) in zip(target_predictions, target_labels, self._metrics.items())
        }

    def _train_epoch(self, epoch, log = True):
        self.model.train()
        step = 0
        for batch_idx, (raw_inputs, symbolic_inputs, _, target_labels) in enumerate(self._train_loader):
            raw_inputs = self._tensor_list_to_device(raw_inputs)
            symbolic_inputs = self._tensor_list_to_device(symbolic_inputs)
            target_labels = self._tensor_list_to_device(target_labels)

            self._perception_optimiser.zero_grad()
            if step % self._config["train"]["perception_steps_per_reasoning_step"] == 0:
                self._reasoning_optimiser.zero_grad()
            target_predictions = self.model(raw_inputs, symbolic_inputs)
            batch_loss = self._compute_loss(target_predictions, target_labels)
            batch_loss.backward()
            if "gradient_clip" in self._config:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._config["train"]["gradient_clip"])
            self._perception_optimiser.step()
            if step % self._config["train"]["perception_steps_per_reasoning_step"] == 0:
                self._reasoning_optimiser.step()
            if self._use_scheduler:
                self._perception_scheduler.step()
            if self._use_scheduler and step % self._config["train"]["perception_steps_per_reasoning_step"] == 0:
                self._reasoning_scheduler.step()
            metrics = self._compute_metrics(target_predictions, target_labels)

            if log:
                self._log("train", {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "batch_loss": batch_loss.item(),
                    **metrics
                })

    @torch.no_grad()
    def validate_epoch(self, epoch, log = True):
        epoch_losses = list()
        epoch_metrics = list()
        self.model.eval()
        for batch_idx, (raw_inputs, symbolic_inputs, _, target_labels) in enumerate(self._val_loader):
            raw_inputs = self._tensor_list_to_device(raw_inputs)
            symbolic_inputs = self._tensor_list_to_device(symbolic_inputs)
            target_labels = self._tensor_list_to_device(target_labels)

            target_predictions = self.model(raw_inputs, symbolic_inputs)
            epoch_losses.append(self._compute_loss(target_predictions, target_labels))
            epoch_metrics.append(self._compute_metrics(target_predictions, target_labels))

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_metrics = {
            target_name: sum([metrics[target_name] for metrics in epoch_metrics]) / len(epoch_metrics)
            for target_name in epoch_metrics[0].keys()
        }

        if log:
            self._log("val", {
                "epoch": epoch,
                "epoch_loss": epoch_loss.item(),
                **epoch_metrics
            })

        return epoch_loss

    def save_model(self, epoch, epoch_loss):
        makedirs(path.join(self._results_dir, "checkpoints"), exist_ok=True)
        torch.save(self.model.state_dict(), path.join(self._results_dir, "checkpoints", f"epoch_{epoch}.pt"))
        if epoch_loss < self._best_loss:
            self._best_loss = epoch_loss
            torch.save(self.model.state_dict(), path.join(self._results_dir, "checkpoints", "best_model.pt"))

def load_best_fully_neural_model(fully_neural_model, results_dir):
    fully_neural_model.load_state_dict(torch.load(path.join(results_dir, "checkpoints", "best_model.pt")))
    return fully_neural_model