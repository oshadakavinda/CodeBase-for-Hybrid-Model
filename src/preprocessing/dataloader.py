from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=0):
    """
    Create PyTorch DataLoaders using torchvision ImageFolder.
    
    Args:
        train_dir: Path to training data (e.g., 'data/normalized/train')
        val_dir: Path to validation data (e.g., 'data/normalized/valid')
        test_dir: Path to test data (e.g., 'data/normalized/test')
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with train_loader, val_loader, test_loader and class info
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    ])
    
    print("Loading datasets using ImageFolder...")
    
    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Classes: {train_dataset.classes}")
    print(f"✓ Class to index: {train_dataset.class_to_idx}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'classes': train_dataset.classes,
        'num_classes': len(train_dataset.classes),
    }

