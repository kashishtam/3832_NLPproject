import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import f1_score

def f1_score_eval(labels, outputs):
    f_sarcasm = f1_score(y_true=labels,y_pred=outputs , average='macro')
    return f_sarcasm
