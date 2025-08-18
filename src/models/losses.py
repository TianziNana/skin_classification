# === Loss Functions and Training Components ===
print("Setting up loss functions and training components...")

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# === Hybrid Loss Functions ===
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class HybridLoss(nn.Module):
    """Hybrid Loss: Focal Loss + Label Smoothing"""
    def __init__(self, alpha=1, gamma=2, smoothing=0.1, focal_weight=0.7):
        super(HybridLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.focal_weight = focal_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        smooth = self.label_smooth_loss(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * smooth

# Create model instance
print("\nCreating model instance...")
model = FairnessAwareSkinClassifier(
    num_classes=2,
    dropout_rate=0.3
).to(device)

# Calculate model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   Model created successfully")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Create loss function
criterion = HybridLoss(
    alpha=class_weights_dict[1]/class_weights_dict[0],  # Using class weight ratio
    gamma=2.0,
    smoothing=0.1,
    focal_weight=0.7
)

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# Create learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,  # 20 epochs
    eta_min=1e-7
)

print(f"   Loss function: Focal Loss + Label Smoothing")
print(f"   Optimizer: AdamW (lr=1e-4)")
print(f"   Scheduler: CosineAnnealingLR")
print("\nReady to start training!")

# === Import training modules ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

print("Training modules imported successfully")
print(f"   PyTorch version: {torch.__version__}")
print(f"   TIMM version: {timm.__version__}")
