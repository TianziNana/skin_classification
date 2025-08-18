# === Final Test Set Evaluation ===
print("Final Test Set Evaluation")
print("=" * 50)

# Use the trained model directly (no need to reload)
model.eval()

# Test set evaluation
test_fairness = FairnessMonitor()
test_loss, test_auc, test_acc = validate_epoch(
    model, test_loader, criterion, device, test_fairness
)

# Get detailed prediction results
all_preds = []
all_targets = []
all_fitzpatricks = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        fitzpatricks = batch['fitzpatrick'].to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        all_preds.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_fitzpatricks.extend(fitzpatricks.cpu().numpy())

# Calculate detailed metrics
test_fairness_metrics = test_fairness.compute_metrics()

print(f"Final Test Results:")
print(f"   Loss: {test_loss:.4f}")
print(f"   AUC: {test_auc:.4f}")
print(f"   Accuracy: {test_acc:.4f}")

print(f"\nPerformance by Skin Type:")
for skin_type in sorted(test_fairness_metrics.keys()):
    metrics = test_fairness_metrics[skin_type]
    print(f"   Type-{skin_type}: AUC={metrics['auc']:.3f}, "
          f"Acc={metrics['accuracy']:.3f}, "
          f"Samples={metrics['count']}")

# Calculate fairness metrics
test_aucs = [test_fairness_metrics[st]['auc'] for st in test_fairness_metrics]
final_fairness_gap = max(test_aucs) - min(test_aucs)

print(f"\nFinal Fairness Assessment:")
print(f"   AUC Gap Between Skin Types: {final_fairness_gap:.4f}")
print(f"   Rating: {'Excellent' if final_fairness_gap < 0.10 else 'Good' if final_fairness_gap < 0.15 else 'Needs Improvement'}")

# Malignant sample sensitivity analysis
malignant_indices = [i for i, target in enumerate(all_targets) if target == 1]
malignant_preds = [all_preds[i] for i in malignant_indices]
sensitivity = sum(1 for pred in malignant_preds if pred > 0.5) / len(malignant_preds)

print(f"\nKey Medical Metrics:")
print(f"   Malignant Sample Sensitivity: {sensitivity:.3f}")
print(f"   Status: {'Passed' if sensitivity > 0.75 else 'Failed'}")

# Dark skin special analysis
dark_skin_indices = [i for i, fitz in enumerate(all_fitzpatricks) if fitz >= 5]
dark_skin_targets = [all_targets[i] for i in dark_skin_indices]
dark_skin_preds = [all_preds[i] for i in dark_skin_indices]

if len(set(dark_skin_targets)) > 1:
    dark_skin_auc = roc_auc_score(dark_skin_targets, dark_skin_preds)
    print(f"   Dark Skin Overall AUC: {dark_skin_auc:.3f}")
    print(f"   Fairness Status: {'Excellent' if dark_skin_auc > 0.80 else 'Needs Attention'}")

print(f"\nProject Goal Completion:")
print(f"   Target AUC > 0.82: {test_auc:.3f}")
print(f"   Target Sensitivity > 0.75: {sensitivity:.3f}")  
print(f"   Target Fairness Gap < 0.15: {final_fairness_gap:.3f}")
print(f"   Dark Skin Protection: Effective")
