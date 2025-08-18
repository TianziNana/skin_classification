# === Fairness Monitor ===
class FairnessMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = {skin_type: [] for skin_type in [1, 2, 3, 4, 5, 6]}
        self.targets = {skin_type: [] for skin_type in [1, 2, 3, 4, 5, 6]}
        
    def update(self, outputs, targets, fitzpatricks):
        preds = torch.softmax(outputs, dim=1)[:, 1]  # Probability of malignancy
        
        for i in range(len(targets)):
            skin_type = int(fitzpatricks[i].item())
            if skin_type in [1, 2, 3, 4, 5, 6]:
                self.predictions[skin_type].append(float(preds[i].item()))
                self.targets[skin_type].append(int(targets[i].item()))
    
    def compute_metrics(self):
        metrics = {}
        for skin_type in [1, 2, 3, 4, 5, 6]:
            if len(self.targets[skin_type]) > 5 and len(set(self.targets[skin_type])) > 1:
                targets = self.targets[skin_type]
                preds = self.predictions[skin_type]
                auc = roc_auc_score(targets, preds)
                acc = accuracy_score(targets, [1 if p > 0.5 else 0 for p in preds])
                metrics[skin_type] = {'auc': auc, 'accuracy': acc, 'count': len(targets)}
        return metrics
