from torchmetrics import Metric
import torch

class MyF1Score(Metric):
    '''
    Macro F1 Score.
        - This metric computes the F1 score for each class and returns the average.
    '''
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')
    
    def update(self, preds, target):
        # preds: (N, C), target: (N,)
        max_preds = preds.argmax(dim=1)

        assert max_preds.shape == target.shape, f"Shape mismatch: {max_preds.shape} vs {target.shape}"

        for c in range(self.num_classes):
            self.tp[c] += ((max_preds == c) & (target == c)).sum()
            self.fp[c] += ((max_preds == c) & (target != c)).sum()
            self.fn[c] += ((max_preds != c) & (target == c)).sum()
            
    def compute(self):
        eps = 1e-6
        
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        
        return f1_score.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # preds: (N, C), target: (N,)
        max_preds = preds.argmax(dim=1)

        assert max_preds.shape == target.shape, f"Shape mismatch: {max_preds.shape} vs {target.shape}"

        correct = (max_preds == target).sum()
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
