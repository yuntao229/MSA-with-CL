import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)
        
class InfoNCE_mul(InfoNCE):
    """
    Calculate multi-label InfoNCE loss based on weight parameters on negative samples.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, batch=None, labels=None):
        if batch is None or labels is None:
            raise ValueError('Batch data or labels is invalid!')
        return info_nce_mul(batch, labels)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        print(query.shape[-1], positive_key.shape[-1])
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    # print("query:", query, "positive_key:", positive_key, "negative_keys:", negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        # print("positive_logit:", positive_logit)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)
            # print("negative_logits:", negative_logits)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        # print("logits:", logits, "labels:", labels)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
        
    init_list = []
    # print(weighted_entropy(logits, labels))
    return F.cross_entropy(logits, labels, reduction=reduction)

def info_nce_mul(batch, labels, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Modified from info_nce.
    # Necessary checking.
    if batch.dim() != 2:
        raise ValueError('<query/batch_data> must have 2 dimensions.')
    
    # Get positive_key, negative_keys and negative_weights from batch data and labels.
    positive_key, negative_keys, negative_weights = choose_samples_similarity(batch, labels)
    
    if len(negative_keys) == 0:
        return 0.0 ### No suitable negative key found.
    
    # Normalize to unit vectors.
    batch_data = torch.tensor([item.cpu().detach().numpy() for item in positive_key], device=batch.device)
    positive_key = torch.tensor([item.cpu().detach().numpy() for item in positive_key], device=batch.device)
    accumulate_loss = 0.0
    for i in range(positive_key.shape[0]):
        # print(len(negative_keys[i]), negative_weights[i])
        if len(negative_keys[i]) == 0:
            continue
        # print("keys:", negative_keys[i])
        negative_key = torch.stack(negative_keys[i])
        negative_weight = torch.tensor(negative_weights[i], device=batch.device)
        query_temp, positive_key_temp, negative_keys_temp, negative_weights_temp = normalize(torch.unsqueeze(batch_data[i], dim=0), torch.unsqueeze(positive_key[i], dim=0), negative_key, negative_weight)
        # print(query_temp.shape, positive_key_temp.shape, negative_keys_temp.shape, negative_weights_temp.shape)
        # Calculate cosine similarity between positive pairs.
        positive_logit = torch.sum(query_temp * positive_key_temp, dim=1, keepdim=True)
        negative_logits = query_temp @ transpose(negative_keys_temp)
        
        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query_temp.device)
        accumulate_loss = accumulate_loss + weighted_entropy(logits / temperature, labels, negative_weights_temp)
        # print("logits_mul:", logits.shape, "labels_mul:", labels.shape)
    return accumulate_loss

# Choose corresponding positive key and negative keys(calculate weight parameters at the same time) from a batch of data
# Each array is considered as a query once.
def choose_samples(batch_data, labels):
    labels = labels.reshape(-1, 1)
    assert labels.shape[0] == batch_data.shape[0], "Mismatch batchsize between data and labels!"
    labels = labels.reshape(1, -1).squeeze()
    ## Query is the original batch data.
    ## Positive key is set as the most similar vector to the query(Not equal, except for no same emotion sample in the whole batch).
    positive_key_list, negative_keys_list, negative_keys_weights = [], [], []
    for i in range(batch_data.shape[0]):
        negative_keys_ret, negative_weights_ret = [], []
        min_distance = 10000.0
        pos_key_index = i
        # print("size:", batch_data.shape[0], 'batchsize:', batch_data.shape)
        for iter in range(labels.shape[-1]):
            if iter != i:
                square = (labels[iter] - labels[i]) * (labels[iter] - labels[i])
                if labels[iter] * labels[i] > 0 and square < min_distance: ### Choose the most similar sample as positive key for the query vector.
                    min_distance = square
                    pos_key_index = iter
                if labels[iter] * labels[i] < 0: ### Choose opposite emotion samples as negative keys while calculating weighted parameters.
                    delta = abs(labels[iter] + labels[i])
                    beta = 1 / (1 + math.exp(-1 * delta))
                    negative_keys_ret.append(batch_data[iter]) ### Add negative sample into negative keys(2 dimensions).
                    negative_weights_ret.append(beta) ### Save weights of negative samples, corresponded by index.
        negative_keys_list.append(negative_keys_ret)
        negative_keys_weights.append(negative_weights_ret)
        if min_distance == 10000.0:
            pos_key_index = i
        else:
            assert pos_key_index != i, "Positive key is equal to query."
        positive_key_list.append(batch_data[pos_key_index])
    assert len(negative_keys_list) == len(negative_keys_weights), "Length of negative samples is not corresponding to negative weights!"
    assert len(negative_keys_list) == len(positive_key_list), "Length of negative groups is not corresponding to positive samples!"
    assert len(positive_key_list) == len(batch_data), "Length of positive samples is not corresponding to batch data!"
    return positive_key_list, negative_keys_list, negative_keys_weights

# Choose corresponding positive key and negative keys(calculate weight parameters at the same time) from a batch of data
# Each array is considered as a query once.
def choose_samples_similarity(batch_data, labels):
    labels = labels.reshape(-1, 1)
    assert labels.shape[0] == batch_data.shape[0], "Mismatch batchsize between data and labels!"
    labels = labels.reshape(1, -1).squeeze()
    ## Query is the original batch data.
    ## Positive key is set as the most similar vector to the query(Not equal, except for no same emotion sample in the whole batch).
    positive_key_list, negative_keys_list, negative_keys_weights = [], [], []
    for i in range(batch_data.shape[0]):
        negative_keys_ret, negative_weights_ret = [], []
        max_similarity = 0.0
        pos_key_index = i
        # print("size:", batch_data.shape[0], 'batchsize:', batch_data.shape)
        for iter in range(labels.shape[-1]):
            if iter != i:
                similarity = torch.cosine_similarity(batch_data[i], torch.unsqueeze(batch_data[iter], 0))[0]
                if similarity < 0:
                    similarity = 1 - similarity
                similarity = similarity / 2
                if labels[iter] * labels[i] > 0 and similarity > max_similarity: ### Choose the most similar sample as positive key for the query vector.
                    max_similarity = similarity
                    pos_key_index = iter
                if labels[iter] * labels[i] < 0: ### Choose opposite emotion samples as negative keys while calculating weighted parameters.
                    beta = similarity
                    negative_keys_ret.append(batch_data[iter]) ### Add negative sample into negative keys(2 dimensions).
                    negative_weights_ret.append(beta) ### Save weights of negative samples, corresponded by index.
        negative_keys_list.append(negative_keys_ret)
        negative_keys_weights.append(negative_weights_ret)
        if max_similarity == 0:
            pos_key_index = i
        else:
            assert pos_key_index != i, "Positive key is equal to query."
        positive_key_list.append(batch_data[pos_key_index])
    assert len(negative_keys_list) == len(negative_keys_weights), "Length of negative samples is not corresponding to negative weights!"
    assert len(negative_keys_list) == len(positive_key_list), "Length of negative groups is not corresponding to positive samples!"
    assert len(positive_key_list) == len(batch_data), "Length of positive samples is not corresponding to batch data!"
    return positive_key_list, negative_keys_list, negative_keys_weights

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def weighted_entropy(inputs, targets, weights=None):
    exp = torch.exp(inputs)
    expr1 = exp.gather(1, targets.unsqueeze(-1)).squeeze()
    expr2 = exp.sum(1)
    if weights is None:
        weights = torch.tensor(np.ones(len(targets)))
    # print("expr1:", expr1, "expr2:", expr2)
    softmax = weights * expr1 / expr2
    # print("softmax:", softmax)
    log = -torch.log(softmax)
    return log.mean()