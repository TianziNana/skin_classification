# === Training Function ===
def train_epoch(model, train_loader, criterion, optimizer, device, fairness_monitor):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        fitzpatricks = batch['fitzpatrick'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Record predictions
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().detach().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        # Update fairness monitoring
        fairness_monitor.update(outputs.detach(), labels, fitzpatricks)
        
        running_loss += loss.item()
        
        # Update progress bar
        if batch_idx % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_acc = accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_preds])
    
    return epoch_loss, epoch_auc, epoch_acc

# === Validation Function ===
def validate_epoch(model, val_loader, criterion, device, fairness_monitor):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            fitzpatricks = batch['fitzpatrick'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            fairness_monitor.update(outputs, labels, fitzpatricks)
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_acc = accuracy_score(all_targets, [1 if p > 0.5 else 0 for p in all_preds])
    
    return epoch_loss, epoch_auc, epoch_acc

# === Start Training ===
print("Starting model training")
print("=" * 60)

NUM_EPOCHS = 15
best_val_auc = 0.0
train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 50)
    
    # Training
    train_fairness = FairnessMonitor()
    train_loss, train_auc, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, train_fairness
    )
    
    # Validation
    val_fairness = FairnessMonitor()
    val_loss, val_auc, val_acc = validate_epoch(
        model, val_loader, criterion, device, val_fairness
    )
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Record history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)
    
    # Calculate fairness metrics
    train_fairness_metrics = train_fairness.compute_metrics()
    val_fairness_metrics = val_fairness.compute_metrics()
    
    # Calculate fairness gap
    val_aucs_by_skin = [val_fairness_metrics[st]['auc'] for st in val_fairness_metrics]
    fairness_gap = max(val_aucs_by_skin) - min(val_aucs_by_skin) if len(val_aucs_by_skin) > 1 else 0
    
    epoch_time = time.time() - epoch_start_time
    
    # Detailed output
    print(f"Training results:")
    print(f"   Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Accuracy: {train_acc:.4f}")
    print(f"Validation results:")
    print(f"   Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Accuracy: {val_acc:.4f}")
    print(f"Fairness analysis:")
    print(f"   AUC gap between skin types: {fairness_gap:.4f}")
    
    # Show AUC by skin type
    print(f"   AUC by skin type: ", end="")
    for skin_type in sorted(val_fairness_metrics.keys()):
        auc = val_fairness_metrics[skin_type]['auc']
        print(f"Type-{skin_type}: {auc:.3f} ", end="")
    print()
    
    print(f"Training parameters:")
    print(f"   Learning rate: {current_lr:.2e} | Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
            'fairness_gap': fairness_gap
        }, 'best_model.pth')
        print(f"Saved best model (AUC: {val_auc:.4f})")
    
    print("=" * 60)

print(f"\nTraining completed! Best validation AUC: {best_val_auc:.4f}")
