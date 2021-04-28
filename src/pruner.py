import copy
import torch.nn as nn
import torch

device = 'cuda:6' if torch.cuda.is_available() else 'cpu'


class Pruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, current_dataset_idx):
        self.model = model
        self.prune_perc = prune_perc

        self.train_bias = False #no bias 
        self.train_bn = False #no batch normaliation
        self.current_masks = None
        self.previous_masks = previous_masks
        self.current_dataset_idx = current_dataset_idx

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # select all prunable weights.
        previous_mask = previous_mask.to(device)
        tensor = weights[previous_mask.eq(self.current_dataset_idx)] #only prune weights for current dataset
        abs_tensor = tensor.abs()

        cutoff_rank = round(self.prune_perc * tensor.numel())
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]

        # remove those weights which are below cutoff and belong to current
        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

        # mask = 1 - remove_mask
        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() // tensor.numel(), weights.numel()))
        return mask

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks."""
        
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        
        self.previous_masks = self.current_masks

        print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_perc))
        for idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask(module.weight.data, self.previous_masks[idx], idx)
                self.current_masks[idx] = mask.to(device)

                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[idx].eq(0)] = 0.0

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks
        for idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[idx]
                if module.weight.grad is not None:
                    # set grads of all weights not belonging to current dataset to 0.
                    module.weight.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0 
                    if not self.train_bias:
                        # biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)

            elif 'BatchNorm2d' in str(type(module)):
                # set grads of batchnorm params to 0.
                if not self.train_bn:
                    if module.weight.grad is not None:
                        module.weight.grad.data.fill_(0)
                        module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks
        for idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.previous_masks[module_idx].to(device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

    def apply_hard_mask(self, dataset_idx, unrelated_tasks): #e.g unrelzated tasks = [2, 5, 6]
        print(unrelated_tasks)
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.previous_masks[module_idx].to(device)
                weight[mask.eq(0)] = 0.0 #set pruned parameters to zero
                for task in unrelated_tasks:
                    weight[mask.eq(task)] = 0.0 #set unrelated parameters to zero
                weight[mask.gt(dataset_idx)] = 0.0
                
    def concat_original_model(self, dataset_idx, original_model): #to recover original values of parameters
        for module_idx, (module, origin_module) in enumerate(zip(self.model.shared.modules(), original_model.shared.modules())):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                origin_weight = origin_module.weight.data
                mask = self.previous_masks[module_idx].to(device)

                weight[mask.ne(dataset_idx)] = origin_weight[mask.ne(dataset_idx)]

    def calc_parameters_per_task(self):
        """
        check number of parameters allocated for each task, only for check
        """
        list_parameter = []
        for data_idx in range(self.current_dataset_idx):
            task_specific_params = 0
            total_params = 0
            for module_idx, module in enumerate(self.model.shared.modules()):
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        mask = self.previous_masks[module_idx]
                        task_specific_params += mask.eq(data_idx + 1).sum()
                        total_params += mask.le(data_idx + 1).sum()
                        print("task {} parameters per layer : {}".format(data_idx + 1, mask.eq(data_idx + 1).sum()))

            print("task {} task-specific total parameters : {}".format(data_idx + 1, task_specific_params))
            print("task {} total parameters : {}".format(data_idx + 1, total_params))
            list_parameter.append(task_specific_params)
        
        return list_parameter

    def initialize_new_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks

        for idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[idx]
                mask[mask.eq(0)] = self.current_dataset_idx #initialize mask

        self.current_masks = self.previous_masks
        # print(self.current_masks) #for check