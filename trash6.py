import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import  src.psga.train.evaluation.functional as F


input = [2,2,2,3,4,5,5,5,5,5]
target = [2,2,2,3,2,1,1,1,1,3]

sklearn_score = cohen_kappa_score(input, target, weights="quadratic")


input = torch.tensor(input)
target = torch.tensor(target)
torch_score = F.cohen_kappa_score(input, target, weights="quadratic").item()


a = 4




