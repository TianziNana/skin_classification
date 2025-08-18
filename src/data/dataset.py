# === Custom Dataset Class ===
class FitzpatrickDataset(Dataset):
    """
    Fitzpatrick17k Dataset Class - Integrated with Fairness-Aware Augmentation
    """
    
    def __init__(self, dataframe, image_folder, augmenter, phase='train'):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.augmenter = augmenter
        self.phase = phase
        
        # Statistical information
        self.num_samples = len(self.df)
        self.num_malignant = self.df['binary_label'].sum()
        self.num_benign = self.num_samples - self.num_malignant
        
        print(f"   {phase.upper()} Dataset:")
        print(f"     Total samples: {self.num_samples:,}")
        print(f"     Malignant: {self.num_malignant:,} ({self.num_malignant/self.num_samples*100:.1f}%)")
        print(f"     Benign: {self.num_benign:,} ({self.num_benign/self.num_samples*100:.1f}%)")
        
        # Skin type distribution
        fitz_dist = self.df['fitzpatrick_scale'].value_counts().sort_index()
        print(f"     Skin type distribution: {dict(fitz_dist)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get sample information
        row = self.df.iloc[idx]
        image_hash = row['md5hash']
        label = row['binary_label']
        fitz_type = row['fitzpatrick_scale']
        is_malignant = bool(label)
        
        # Load image
        image = self._load_image(image_hash)
        
        # Use fairness-aware augmentation strategy
        transform = self.augmenter.get_augmentation_pipeline(
            fitzpatrick_type=fitz_type,
            is_malignant=is_malignant,
            phase=self.phase
        )
        
        # Apply transformations
        if transform:
            # Convert PIL to numpy (required by albumentations)
            image_np = np.array(image)
            transformed = transform(image=image_np)
            image_tensor = transformed['image']
        else:
            # If no transform, manually convert
            image_np = np.array(image)
            image_tensor = torch.FloatTensor(image_np).permute(2, 0, 1) / 255.0
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'fitzpatrick': torch.tensor(fitz_type, dtype=torch.long),
            'hash': image_hash
        }
    
    def _load_image(self, image_hash):
        """Load image file"""
        # Try different file extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(self.image_folder, image_hash + ext)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    return image
                except Exception as e:
                    continue
        
        # If image not found, return black placeholder
        print(f"Warning: Image not found: {image_hash}")
        return Image.new('RGB', (224, 224), (0, 0, 0))

print("FitzpatrickDataset class created successfully")
print("   - Integrated fairness-aware augmentation strategy")
print("   - Automatically selects augmentation intensity based on skin type and lesion type")

# === Create Dataset Instances ===
print("Creating dataset instances...")

# Initialize augmenter
augmenter = FairnessAwareAugmentation(image_size=224)

# Create dataset instances
train_dataset = FitzpatrickDataset(
    dataframe=train_df, 
    image_folder=image_folder, 
    augmenter=augmenter, 
    phase='train'
)

val_dataset = FitzpatrickDataset(
    dataframe=val_df, 
    image_folder=image_folder, 
    augmenter=augmenter, 
    phase='val'
)

test_dataset = FitzpatrickDataset(
    dataframe=test_df, 
    image_folder=image_folder, 
    augmenter=augmenter, 
    phase='test'
)

print("\nDataset instances created successfully")
print(f"   Training set: {len(train_dataset):,} samples")
print(f"   Validation set: {len(val_dataset):,} samples") 
print(f"   Test set: {len(test_dataset):,} samples")
