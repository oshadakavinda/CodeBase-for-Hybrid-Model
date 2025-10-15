import torch
import torch.nn as nn

def train_rnn(model, train_loader, val_loader, device, optimizer, criterion, epochs=50):
    """Train the RNN model with cross-entropy loss for classification.
    
    Args:
        model: RNNModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on (cuda or cpu)
        optimizer: Optimizer for training
        criterion: Loss function (CrossEntropyLoss)
        epochs: Number of training epochs
    
    Returns:
        model: Trained model
        history: Dictionary with training/validation metrics
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            # Reshape images for RNN: (batch_size, 1, 80, 80) -> (batch_size, 80, 80)
            images = images.squeeze(1)  # Remove channel dimension
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.squeeze(1)  # Remove channel dimension
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return model, history

# Usage example
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assuming you have: rnn_model, train_loader, val_loader, optimizer_rnn defined
    criterion = nn.CrossEntropyLoss()
    
    rnn_model, history = train_rnn(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer_rnn,
        criterion=criterion,
        epochs=50
    )
    
    print("\nTraining completed!")
    print(f"Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"Final Val Acc: {history['val_acc'][-1]:.2f}%")