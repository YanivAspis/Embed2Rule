import torch


def categorical_accuracy(predictions, labels):
    corrects = torch.sum(torch.argmax(predictions, dim=1) == torch.argmax(labels, dim=1))
    return corrects / len(labels)

def binary_accuracy(predictions, labels):
    corrects = torch.sum(torch.round(predictions) == labels)
    return corrects / len(labels)