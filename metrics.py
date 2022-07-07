import torch
from torch import nn
from torch.nn import functional as F
from scipy import optimize
import numpy as np

from tqdm import tqdm

def bin_confidences_and_accuracies(confidences, ground_truth, bin_edges, indices):
    i = np.arange(0, bin_edges.size-1)
    aux = indices == i.reshape((-1, 1))
    counts = aux.sum(axis=1)
    weights = counts / np.sum(counts)
    correct = np.logical_and(aux, ground_truth).sum(axis=1)
    a = np.repeat(confidences.reshape(1, -1), bin_edges.size-1, axis=0)
    a[np.logical_not(aux)] = 0
    bin_accuracy = correct / counts
    bin_confidence = a.sum(axis=1) / counts
    return weights, bin_accuracy, bin_confidence


def get_ece(confidences, ground_truth, nbins):
    # Repeated code from determine edges. Here it is okay if the bin edges are not unique defined
    confidences_sorted = confidences.copy()
    confidences_index = confidences.argsort()
    confidences_sorted = confidences_sorted[confidences_index]
    aux = np.linspace(0, len(confidences_sorted) - 1, nbins + 1).astype(int) + 1
    bin_indices = np.zeros(len(confidences_sorted)).astype(int)
    bin_indices[:aux[1]] = 0
    for i in range(1, len(aux) - 1):
        bin_indices[aux[i]:aux[i + 1]] = i
    bin_edges = np.zeros(nbins + 1)
    for i in range(0, nbins - 1):
        bin_edges[i + 1] = np.mean(np.concatenate((
            confidences_sorted[bin_indices == i][confidences_sorted[bin_indices == i] == max(confidences_sorted[bin_indices == i])],
            confidences_sorted[bin_indices == (i + 1)][
                confidences_sorted[bin_indices == (i + 1)] == min(confidences_sorted[bin_indices == (i + 1)])])))
    bin_edges[0] = 0
    bin_edges[-1] = 1
    bin_indices = bin_indices[np.argsort(confidences_index)]

    weights, bin_accuracy, bin_confidence = bin_confidences_and_accuracies(confidences, ground_truth, bin_edges,
                                                                           bin_indices)
    ece = np.dot(weights, np.abs(bin_confidence - bin_accuracy))
    return ece


def get_metrics(softmaxes, labels):
    results = {}
    
    # Compute Accuracy
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    results['acc'] = accuracies.float().mean().item()

    # Top-1 ECE equal mass (with nbins=15)
    confidences = confidences.numpy()
    accuracies = accuracies.numpy()
    results['top1_ece_eq_mass'] = get_ece(confidences, accuracies, 15)
        
    return results