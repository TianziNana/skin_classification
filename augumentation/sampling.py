import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
import glob

# Set paths
csv_path = '/kaggle/input/fitzpatrick17k/New folder/fitzpatrick17k (1).csv'
image_folder = '/kaggle/input/fitzpatrick17k/New folder/background removed'

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

def load_and_prepare_data():
    """Load CSV data and prepare samples"""
    try:
        # Load CSV file
        print(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV file loaded successfully, total rows: {len(df)}")
        
        # Show basic CSV info
        print("\nCSV file columns:")
        print(df.columns.tolist())
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Check image folder
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder does not exist: {image_folder}")
            return None
            
        print(f"\nImage folder exists: {image_folder}")
        
        # Show Fitzpatrick type distribution
        if 'fitzpatrick_scale' in df.columns:
            print(f"\nFitzpatrick type distribution:")
            print(df['fitzpatrick_scale'].value_counts().sort_index())
        
        # Show malignant/benign distribution
        malignant_cols = ['malignant', 'binary_label', 'diagnosis']
        malignant_col = None
        for col in malignant_cols:
            if col in df.columns:
                malignant_col = col
                break
                
        if malignant_col:
            print(f"\n{malignant_col} distribution:")
            print(df[malignant_col].value_counts())
        
        return df, malignant_col
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def find_image_file(image_identifier, image_folder):
    """Find image file based on identifier"""
    # Possible image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for ext in extensions:
        # Try different naming patterns
        possible_names = [
            f"{image_identifier}{ext}",
            f"{image_identifier}{ext.upper()}",
        ]
        
        for name in possible_names:
            full_path = os.path.join(image_folder, name)
            if os.path.exists(full_path):
                return full_path
    
    return None

def load_image_safely(image_path):
    """Safely load image"""
    try:
        if not os.path.exists(image_path):
            print(f"Warning: Image file does not exist: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image: {image_path}")
            return None
            
        # Convert color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def calculate_image_metrics(original, augmented):
    """Calculate image quality metrics between original and augmented images"""
    
    # Convert to grayscale for some metrics
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
    
    # Calculate metrics
    ssim_score = ssim(orig_gray, aug_gray, data_range=255)
    psnr_score = psnr(orig_gray, aug_gray, data_range=255)
    
    # Histogram analysis
    orig_hist = cv2.calcHist([original], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    aug_hist = cv2.calcHist([augmented], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_correlation = cv2.compareHist(orig_hist, aug_hist, cv2.HISTCMP_CORREL)
    
    # Color distribution analysis
    orig_mean = np.mean(original, axis=(0, 1))
    aug_mean = np.mean(augmented, axis=(0, 1))
    color_shift = np.linalg.norm(orig_mean - aug_mean)
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score,
        'hist_corr': hist_correlation,
        'color_shift': color_shift
    }

def get_samples_for_analysis(df, malignant_col, num_samples=4):
    """Select representative samples from dataset for analysis"""
    samples = []
    
    # Define desired sample types
    target_cases = [
        {'fitzpatrick': [1, 2], 'malignant': 0, 'name': 'Light_Benign'},
        {'fitzpatrick': [1, 2], 'malignant': 1, 'name': 'Light_Malignant'},
        {'fitzpatrick': [5, 6], 'malignant': 0, 'name': 'Dark_Benign'},
        {'fitzpatrick': [5, 6], 'malignant': 1, 'name': 'Dark_Malignant'}
    ]
    
    for case in target_cases[:num_samples]:
        # Filter matching samples
        mask = df['fitzpatrick_scale'].isin(case['fitzpatrick'])
        if malignant_col:
            mask = mask & (df[malignant_col] == case['malignant'])
        
        filtered_df = df[mask]
        
        if len(filtered_df) > 0:
            # Randomly select one sample
            sample = filtered_df.sample(n=1).iloc[0]
            samples.append({
                'data': sample,
                'fitzpatrick': sample['fitzpatrick_scale'],
                'malignant': case['malignant'],
                'name': case['name']
            })
            print(f"Found {case['name']} sample: Fitzpatrick={sample['fitzpatrick_scale']}")
        else:
            print(f"No {case['name']} samples found")
    
    return samples

def visualize_augmentation_comparison():
    """Generate augmentation comparison using real data from CSV"""
    
    # Load data
    data_result = load_and_prepare_data()
    if data_result is None:
        return None
        
    df, malignant_col = data_result
    
    # Get analysis samples
    samples = get_samples_for_analysis(df, malignant_col)
    
    if not samples:
        print("No suitable samples found for analysis")
        return None
    
    augmenter = FairnessAwareAugmentation(image_size=224)
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Fairness-Aware Augmentation Comparison', fontsize=16)
    
    metrics_summary = []
    
    for i, sample in enumerate(samples):
        sample_data = sample['data']
        
        # Determine image filename
        # Try different possible column names
        image_id_cols = ['md5hash', 'image_id', 'filename', 'image_name']
        image_identifier = None
        
        for col in image_id_cols:
            if col in sample_data.index and pd.notna(sample_data[col]):
                image_identifier = sample_data[col]
                break
        
        if image_identifier is None:
            print(f"Cannot determine image identifier for sample {i+1}")
            continue
            
        # Find image file
        image_path = find_image_file(image_identifier, image_folder)
        if image_path is None:
            print(f"Image file not found: {image_identifier}")
            continue
            
        print(f"\nProcessing sample {i+1}: {os.path.basename(image_path)}")
        
        # Load original image
        original_image = load_image_safely(image_path)
        if original_image is None:
            print(f"Skipping image: {image_path}")
            continue
            
        # Resize image
        original_image = cv2.resize(original_image, (224, 224))
        
        # Apply augmentation
        transform = augmenter.get_augmentation_pipeline(
            fitzpatrick_type=sample['fitzpatrick'],
            is_malignant=bool(sample['malignant']),
            phase='train'
        )
        
        try:
            # Create augmented version
            augmented = transform(image=original_image)['image']
            
            # Denormalize for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            augmented_np = augmented.permute(1, 2, 0).numpy()
            augmented_np = (augmented_np * std + mean) * 255
            augmented_image = np.clip(augmented_np, 0, 255).astype(np.uint8)
            
            # Calculate intensity factor for reference
            intensity = augmenter.get_intensity_factor(sample['fitzpatrick'], bool(sample['malignant']))
            
            # Calculate metrics
            metrics = calculate_image_metrics(original_image, augmented_image)
            metrics['case'] = sample['name']
            metrics['intensity_factor'] = intensity
            metrics['image_file'] = os.path.basename(image_path)
            metrics['fitzpatrick'] = sample['fitzpatrick']
            metrics['malignant'] = sample['malignant']
            metrics_summary.append(metrics)
            
            # Plot comparison
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original: {sample['name']}\nFitzpatrick={sample['fitzpatrick']}, Malignant={sample['malignant']}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(augmented_image)
            axes[i, 1].set_title(f"Augmented (x{intensity:.1f})")
            axes[i, 1].axis('off')
            
            # Show difference
            diff = cv2.absdiff(original_image, augmented_image)
            axes[i, 2].imshow(diff)
            axes[i, 2].set_title(f"Difference")
            axes[i, 2].axis('off')
            
            print(f"  {sample['name']}:")
            print(f"    Fitzpatrick Type: {sample['fitzpatrick']}")
            print(f"    Malignant: {sample['malignant']}")
            print(f"    Intensity Factor: {intensity:.2f}")
            print(f"    SSIM: {metrics['ssim']:.3f}")
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
            print(f"    Histogram Correlation: {metrics['hist_corr']:.3f}")
            print(f"    Color Shift: {metrics['color_shift']:.2f}")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue
    
    plt.tight_layout()
    plt.show()
    
    return metrics_summary

def plot_metrics_summary(metrics_summary):
    """Plot summary of augmentation metrics"""
    if not metrics_summary:
        print("No valid metrics data to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Augmentation Metrics Summary', fontsize=16)
    
    cases = [f"{m['case']}\nF{m['fitzpatrick']}_M{m['malignant']}" for m in metrics_summary]
    
    # SSIM scores
    ssim_scores = [m['ssim'] for m in metrics_summary]
    axes[0, 0].bar(range(len(cases)), ssim_scores, color='skyblue')
    axes[0, 0].set_title('SSIM Scores')
    axes[0, 0].set_ylabel('SSIM')
    axes[0, 0].set_xticks(range(len(cases)))
    axes[0, 0].set_xticklabels(cases, rotation=45, ha='right')
    
    # PSNR scores
    psnr_scores = [m['psnr'] for m in metrics_summary]
    axes[0, 1].bar(range(len(cases)), psnr_scores, color='lightgreen')
    axes[0, 1].set_title('PSNR Scores')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_xticks(range(len(cases)))
    axes[0, 1].set_xticklabels(cases, rotation=45, ha='right')
    
    # Histogram correlation
    hist_corr = [m['hist_corr'] for m in metrics_summary]
    axes[1, 0].bar(range(len(cases)), hist_corr, color='lightcoral')
    axes[1, 0].set_title('Histogram Correlation')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_xticks(range(len(cases)))
    axes[1, 0].set_xticklabels(cases, rotation=45, ha='right')
    
    # Intensity factors
    intensity_factors = [m['intensity_factor'] for m in metrics_summary]
    axes[1, 1].bar(range(len(cases)), intensity_factors, color='gold')
    axes[1, 1].set_title('Augmentation Intensity Factors')
    axes[1, 1].set_ylabel('Intensity Factor')
    axes[1, 1].set_xticks(range(len(cases)))
    axes[1, 1].set_xticklabels(cases, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Execute comparison
    print("Starting augmentation analysis...")
    print("Using real Fitzpatrick17k dataset")
    print("="*50)
    
    metrics_results = visualize_augmentation_comparison()
    
    if metrics_results:
        print("\n" + "="*50)
        print("Plotting metrics summary...")
        plot_metrics_summary(metrics_results)
        
        print(f"\nAnalysis summary:")
        print(f"- Successfully processed {len(metrics_results)} real medical images")
        print("- Darker skin types received higher intensity factors")
        print("- Malignant samples received additional augmentation protection")
        print("- Dark skin malignant samples received the highest intensity augmentation")
        print("- SSIM and PSNR scores indicate augmentation quality")
        print("- Fairness-aware augmentation helps reduce model bias")
    else:
        print("No images were successfully processed")

print("\nFairness-aware augmenter created successfully")
print("   - Uses real Fitzpatrick17k dataset")
print("   - Adjusts augmentation intensity based on skin type and lesion nature")
print("   - Dark skin malignant samples receive highest protection")
print("   - Includes 15 medical image augmentation techniques")
