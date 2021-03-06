import torch
import torch.nn.functional as F
from .consistency import ConsistencyRegularization
from ..consistency.cross_entropy import CrossEntropy
from .utils import make_pseudo_label, sharpening

class PseudoLabel(ConsistencyRegularization):
    """
    PseudoLabel

    Parameters
    --------
    consistency: str
        consistency objective name
    threshold: float
        threshold to make mask
    sharpen: float
        sharpening temperature for target value
    temp_softmax: float
        temperature for temperature softmax
    """
    def __init__(
        self,
        consistency,
        threshold = 0.95,
        sharpen: float = None,
        temp_softmax: float = None,
        num_classes: int = 10,
    ):
        super().__init__(
            consistency,
            threshold,
            sharpen,
            temp_softmax
        )
        self.num_classes = num_classes

    def __call__(self, stu_preds, tea_logits, *args, **kwargs):
        hard_label, mask = make_pseudo_label(tea_logits, self.threshold)
        #onehot_label = self.__make_one_hot(hard_label,num_classes=self.num_classes).float()
        return stu_preds, hard_label, mask
    
    def __make_one_hot(self, y, num_classes=10):
        return torch.eye(num_classes)[y].to(y.device)
    def __repr__(self):
        return f"PseudoLabel(threshold={self.threshold}, sharpen={self.sharpen}, tau={self.tau})"

