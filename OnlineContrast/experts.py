import numpy as np
import csv
import os
import torch
import torch.nn.functional as F

STU_NAME = 'student'

def confidence(gram, temperature=0.1):
    device = (torch.device('cuda')
              if torch.cuda.is_available()
              else torch.device('cpu'))

    # compute logits
    anchor_dot_contrast = torch.div(
        gram,
        temperature)  # (2*bsz, 2*bsz)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    batch_size = int(gram.shape[0] / 2)
    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    mask = mask.repeat(2, 2)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * 2).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    confidence = (mask * exp_logits).sum(1) / exp_logits.sum(1)

    return confidence.mean()


class MultiWeightedUpdate(object):
    def __init__(self, opt, model_list):
        self.tch_model_names = model_list
        self.tch_model_len = len(model_list)

        # Add student model to model list
        model_list += [STU_NAME]
        self.total_model_names = model_list
        self.total_model_len = len(model_list)

        # Initialization for confidence records
        self.cur_len = 0
        self.log_conf_file = os.path.join(opt.save_folder, 'confidence.csv')
        self.log_weights_file = os.path.join(opt.save_folder, 'weights.csv')
        self.log_cnt_file = os.path.join(opt.save_folder, 'best_tch.csv')
        self.records = [[] for _ in range(self.total_model_len)]

        # Initialization for multiplicative weighted algorithm
        self.weight_update = opt.weight_update
        self.alpha = opt.alpha
        self.eps = opt.epsilon
        self.weights = np.ones(self.tch_model_len)

    def add_record(self, model_name, confidence):
        assert model_name in self.total_model_names, \
            'model name {} does not exist!'.format(model_name)

        ind = self.total_model_names.index(model_name)
        self.records[ind].append(confidence)

        # Update the weight according to moving average
        if model_name != STU_NAME:
            if self.weight_update == 'equal':  # Do nothing, keep equal weights of 1
                pass

            elif self.weight_update == 'single':  # Keep the current confidence
                self.weights[ind] = confidence

            elif self.weight_update == 'ma':  # Moving-average policy
                if self.weights[ind] <= 0:  # first time
                    self.weights[ind] = confidence
                else:
                    self.weights[ind] = self.weights[ind] * self.alpha + \
                                            confidence * (1 - self.alpha)

            elif self.weight_update == 'mw':  # Multiplicative weight policy
                self.weights[ind] *= (1 + self.eps * confidence)

            else:
                raise ValueError('Experts weight update {} not supported!'.format(self.weight_update))

    def get_weight(self, model_name):
        # Generate the distribution based on weights
        dist = self.weights / self.weights.sum()
        ind = self.tch_model_names.index(model_name)

        if self.weight_update == 'ma' or self.weight_update == 'equal':
            return dist[ind]

        else:  # self.weight_update == 'single'
            return float(dist[ind] == max(dist))

    def get_normed_weights(self):
        dist = self.weights / self.weights.sum()

        if self.weight_update == 'ma' or self.weight_update == 'equal':
            return dist

        else:  # self.weight_update == 'single'
            return (np.array(dist) == max(dist)).astype('float')


    def log_once(self):
        # log the new record at cur_len
        with open(self.log_conf_file, 'a+') as outcsv:
            writer = csv.writer(outcsv, delimiter=',')
            if self.cur_len == 0:
                writer.writerow(self.total_model_names)
            writer.writerow([r[self.cur_len] for r in self.records])

        with open(self.log_weights_file, 'a+') as outcsv:
            writer = csv.writer(outcsv, delimiter=',')
            if self.cur_len == 0:
                writer.writerow(self.tch_model_names)
            writer.writerow(self.weights)

        self.cur_len += 1

