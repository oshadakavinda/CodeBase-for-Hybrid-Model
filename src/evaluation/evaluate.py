import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, device):
    """Evaluate classification model on test set.
    
    Args:
        model: Trained model (CNNModel, RNNModel, or HybridModel)
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        dict: Metrics (accuracy, precision, recall, f1)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Ensure output is 2D (batch_size, num_classes)
            if outputs.dim() == 3:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(test_loader)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss
    }
    
    return metrics, all_preds, all_labels

def print_evaluation_report(model, test_loader, model_name, device):
    """Print detailed evaluation report.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        model_name: Name of model (e.g., 'CNN', 'RNN', 'Hybrid')
        device: Device to run evaluation on
    """
    metrics, preds, labels = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Report: {model_name} Model")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"{'='*50}\n")
    
    return metrics

def compare_models(models_dict, test_loader, device):
    """Compare multiple models on test set.
    
    Args:
        models_dict: Dictionary with model names as keys and models as values
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    """
    results = {}
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for model_name, model in models_dict.items():
        metrics, _, _ = evaluate_model(model, test_loader, device)
        results[model_name] = metrics
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
    
    print("\n" + "="*60)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics[m]*100 for m in metrics_names]
        ax.bar(x + i*width, values, width, label=model_name)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results