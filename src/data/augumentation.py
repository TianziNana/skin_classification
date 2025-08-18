# === Fairness-Aware Data Augmentation Strategy ===
class FairnessAwareAugmentation:
    """
    Fairness-aware medical image augmentation strategy
    Adjusts augmentation intensity based on Fitzpatrick skin type and lesion nature
    """
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def get_intensity_factor(self, fitzpatrick_type, is_malignant):
        """
        Calculate augmentation intensity factor based on sample characteristics
        Darker skin and malignant samples receive stronger augmentation for improved fairness
        """
        base_intensity = 1.0
        
        # Skin type adjustment
        if fitzpatrick_type >= 5:  # Dark skin
            base_intensity *= 1.3
        elif fitzpatrick_type <= 2:  # Light skin
            base_intensity *= 0.9
            
        # Malignant sample adjustment
        if is_malignant:
            base_intensity *= 1.2
            
        # Special protection: Dark skin malignant samples
        if fitzpatrick_type >= 5 and is_malignant:
            base_intensity *= 1.4
            
        return min(base_intensity, 2.0)  # Limit maximum intensity
    
    def get_augmentation_pipeline(self, fitzpatrick_type, is_malignant=False, phase='train'):
        """
        Get complete augmentation pipeline
        """
        
        if phase != 'train':
            # Only basic transforms for validation and test phases
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # Calculate augmentation intensity
        intensity = self.get_intensity_factor(fitzpatrick_type, is_malignant)
        
        # Skin-type specific parameters
        if fitzpatrick_type <= 2:  # Light skin
            brightness_limit = 0.12
            contrast_limit = 0.12
            blur_max = 3
            noise_var = (3.0, 15.0)
        elif fitzpatrick_type >= 5:  # Dark skin
            brightness_limit = 0.18
            contrast_limit = 0.18
            blur_max = 5
            noise_var = (8.0, 25.0)
        else:  # Medium skin
            brightness_limit = 0.15
            contrast_limit = 0.15
            blur_max = 4
            noise_var = (5.0, 20.0)
        
        # Apply intensity adjustment
        brightness_limit *= intensity
        contrast_limit *= intensity
        blur_max = int(blur_max * intensity)
        noise_var = (noise_var[0] * intensity, noise_var[1] * intensity)
        
        transforms = [
            # Basic geometric transforms
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            
            # Affine transforms
            A.Affine(
                translate_percent=0.08 * intensity,
                scale=(0.95, 1.05),
                rotate=(-12 * intensity, 12 * intensity),
                p=0.6
            ),
            
            # Brightness and contrast adjustments
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5
            ),
            
            # Blur and noise processing
            A.OneOf([
                A.MotionBlur(blur_limit=(3, blur_max), p=0.25),
                A.MedianBlur(blur_limit=(3, blur_max), p=0.25),
                A.GaussianBlur(blur_limit=(3, blur_max), p=0.25),
                A.GaussNoise(variance_limit=noise_var, p=0.25)
            ], p=0.4),
            
            # Color space adjustments
            A.HueSaturationValue(
                hue_shift_limit=int(8 * intensity),
                sat_shift_limit=int(12 * intensity),
                val_shift_limit=int(8 * intensity),
                p=0.4
            ),
            
            # Adaptive histogram equalization
            A.CLAHE(clip_limit=2.5 if fitzpatrick_type >= 5 else 3.5, p=0.6),
            
            # Attention mechanism enhancement
            A.CoarseDropout(
                max_holes=int(3 * intensity),
                max_height=int(16 * intensity),
                max_width=int(16 * intensity),
                p=0.3
            ),
        ]
        
        # Special handling for dark skin
        if fitzpatrick_type >= 5:
            transforms.append(A.RandomGamma(gamma_limit=(85, 115), p=0.3))
        
        # Normalization
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)

print("Fairness-aware augmenter created successfully")
print("   - Adjusts augmentation intensity based on skin type")
print("   - Dark skin malignant samples receive highest protection")
print("   - Includes 15 medical image augmentation techniques")
