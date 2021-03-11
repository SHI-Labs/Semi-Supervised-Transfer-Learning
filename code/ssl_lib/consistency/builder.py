from .cross_entropy import CrossEntropy
from .mean_squared import MeanSquared


def gen_consistency(type,use_onehot=False,num_classes = 10):
    if type == "ce":
        return CrossEntropy()
    elif type == "ms":
        return MeanSquared(use_onehot=use_onehot,num_classes = num_classes)
    else:
        return None