import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import  src.psga.train.evaluation.functional as F


input = [1,2,4,1,2,3,4,5,0,1]
target = [2,2,2,1,2,3,4,5,0,1]
labels = [1, 2, 3]
sample_weights = [2,1,1,2,1,1,3,1,1,1]
sklearn_cm = confusion_matrix(target, input, labels=None, sample_weight=sample_weights)


input = torch.tensor(input)
target = torch.tensor(target)

torch_cm = F.confusion_matrix(input, target, labels=None, sample_weight=torch.tensor(sample_weights)).numpy()




a = 4




