import torch
import torch.nn.functional as F
import editdistance
import matplotlib.pyplot as plt

def compute_cer(preds, labels, char_to_idx, blank_idx=0):
    """Compute Character Error Rate (CER) for CTC-based predictions.
    
    Args:
        preds: Model outputs, shape (batch_size, seq_len, num_classes).
        labels: Ground truth labels, shape (batch_size, seq_len) or (batch_size,).
        char_to_idx: Dictionary mapping characters to indices.
        blank_idx: Index of the blank token (default 0).
    
    Returns:
        float: Average CER across the batch.
    """
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    idx_to_char[blank_idx] = ''
    total_cer = 0
    batch_size = preds.size(0)
    
    for i in range(batch_size):
        pred = preds[i].argmax(dim=1)
        pred_str = ''.join(idx_to_char.get(idx.item(), '') for idx in pred if idx.item() != blank_idx)
        
        if labels.dim() == 2:
            true_str = ''.join(idx_to_char.get(idx.item(), '') for idx in labels[i] if idx.item() != blank_idx)
        else:
            true_str = idx_to_char.get(labels[i].item(), '')
        
        cer = editdistance.eval(pred_str, true_str) / max(len(true_str), 1)  # Use editdistance.eval
        total_cer += cer
    
    return total_cer / batch_size

def compute_wer(preds, labels, char_to_idx, blank_idx=0):
    """Compute Word Error Rate (WER). Treats each character sequence as a word.
    
    Args:
        preds: Model outputs, shape (batch_size, seq_len, num_classes).
        labels: Ground truth labels, shape (batch_size, seq_len) or (batch_size,).
        char_to_idx: Dictionary mapping characters to indices.
        blank_idx: Index of the blank token (default 0).
    
    Returns:
        float: Average WER across the batch.
    """
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    idx_to_char[blank_idx] = ''
    total_wer = 0
    batch_size = preds.size(0)
    
    for i in range(batch_size):
        pred = preds[i].argmax(dim=1)
        pred_str = ''.join(idx_to_char.get(idx.item(), '') for idx in pred if idx.item() != blank_idx)
        
        if labels.dim() == 2:
            true_str = ''.join(idx_to_char.get(idx.item(), '') for idx in labels[i] if idx.item() != blank_idx)
        else:
            true_str = idx_to_char.get(labels[i].item(), '')
        
        wer = 1.0 if pred_str != true_str else 0.0
        total_wer += wer
    
    return total_wer / batch_size

def evaluate_hybrid(model, test_loader, char_to_idx, device):
    """Evaluate the HybridModel using CER, WER, and accuracy.
    
    Args:
        model: Trained HybridModel.
        test_loader: DataLoader for test data.
        char_to_idx: Dictionary mapping characters to indices.
        device: Device to run evaluation on.
    
    Returns:
        tuple: (CER, WER, accuracy).
    """
    model.eval()
    total_cer, total_wer, correct, total = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            print(f"Evaluate HybridModel - Output shape: {outputs.shape}, Label shape: {labels.shape}")
            
            outputs = F.log_softmax(outputs, dim=2)
            total_cer += compute_cer(outputs, labels, char_to_idx, blank_idx=0)
            total_wer += compute_wer(outputs, labels, char_to_idx, blank_idx=0)
            
            if labels.dim() == 1:
                preds = outputs.argmax(dim=2)[:, 0]
                correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    cer = total_cer / len(test_loader)
    wer = total_wer / len(test_loader)
    accuracy = correct / total if total > 0 else 0.0
    return cer, wer, accuracy

def evaluate_cnn(model, test_loader, char_to_idx, device):
    """Evaluate the CNNModel using accuracy.
    
    Args:
        model: Trained CNNModel.
        test_loader: DataLoader for test data.
        char_to_idx: Dictionary mapping characters to indices.
        device: Device to run evaluation on.
    
    Returns:
        float: Accuracy.
    """
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            print(f"Evaluate CNNModel - Output shape: {outputs.shape}, Label shape: {labels.shape}")
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_rnn(model, test_loader, char_to_idx, device):
    """Evaluate the RNNModel using CER, WER, and accuracy.
    
    Args:
        model: Trained RNNModel.
        test_loader: DataLoader for test data.
        char_to_idx: Dictionary mapping characters to indices.
        device: Device to run evaluation on.
    
    Returns:
        tuple: (CER, WER, accuracy).
    """
    model.eval()
    total_cer, total_wer, correct, total = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            print(f"Evaluate RNNModel - Output shape: {outputs.shape}, Label shape: {labels.shape}")
            
            outputs = F.log_softmax(outputs, dim=2)
            total_cer += compute_cer(outputs, labels, char_to_idx, blank_idx=0)
            total_wer += compute_wer(outputs, labels, char_to_idx, blank_idx=0)
            
            if labels.dim() == 1:
                preds = outputs.argmax(dim=2)[:, 0]
                correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    cer = total_cer / len(test_loader)
    wer = total_wer / len(test_loader)
    accuracy = correct / total if total > 0 else 0.0
    return cer, wer, accuracy

def visualize_predictions(model, test_loader, char_to_idx, device, num_samples=5, save_path="outputs/visualizations/predictions.png"):
    """Visualize model predictions on a batch of test data.
    
    Args:
        model: Trained model (HybridModel, CNNModel, or RNNModel).
        test_loader: DataLoader for test data.
        char_to_idx: Dictionary mapping characters to indices.
        device: Device to run the model on.
        num_samples: Number of samples to visualize (default: 5).
        save_path: Path to save the visualization.
    """
    model.eval()
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    idx_to_char[0] = ''
    
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        if outputs.dim() == 3:
            preds = outputs.argmax(dim=2)
            pred_strings = [
                ''.join(idx_to_char.get(idx.item(), '') for idx in pred if idx.item() != 0)
                for pred in preds
            ][:num_samples]
        else:
            preds = outputs.argmax(dim=1)
            pred_strings = [idx_to_char.get(pred.item(), '') for pred in preds][:num_samples]
        
        if labels.dim() == 2:
            true_strings = [
                ''.join(idx_to_char.get(idx.item(), '') for idx in label if idx.item() != 0)
                for label in labels
            ][:num_samples]
        else:
            true_strings = [idx_to_char.get(label.item(), '') for label in labels][:num_samples]
        
        fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img.squeeze(2)
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(f"Pred: {pred_strings[i]}\nTrue: {true_strings[i]}")
            ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()