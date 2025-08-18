# === Phase 3: Model Construction and Training ===
import torch.nn as nn
import torch.nn.functional as F
import timm  # For EfficientNet

print("Phase 3: Model Construction and Training")
print("=" * 50)

# === Fairness-Aware Skin Disease Classifier ===
class FairnessAwareSkinClassifier(nn.Module):
    """
    Fairness-aware skin disease classifier
    Binary classifier based on EfficientNet-B3 architecture
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(FairnessAwareSkinClassifier, self).__init__()
        
        # Using EfficientNet-B3 as backbone [6,7](@ref)
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension (1536 for B3) [6](@ref)
        self.feature_dim = self.backbone.num_features
        
        # Classification head design with progressive dimensionality reduction [1,5](@ref)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(256, num_classes)
        )
        
        # Feature hook for fairness analysis [4](@ref)
        self.features = None
        
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        self.features = features  # Save features for analysis
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x):
        """Extract image feature vectors"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

print("FairnessAwareSkinClassifier model class definition completed")
print("   - Based on EfficientNet-B3 pretrained model")
print("   - Three-layer classification head with progressive dimensionality reduction")
print("   - Integrated feature extraction for fairness analysis")
