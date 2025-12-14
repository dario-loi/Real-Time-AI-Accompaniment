import sys
import os

# Add project root to Python path (critical for imports when running from nested directories)
# __file__ gives us the path to THIS file, then we go up 3 directories to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler


from src.utils.logger import setup_logger
from src.model.lstm_model import ChordLSTM
from src.model.vocabulary import Vocabulary
from src.model.train.dataloader import create_loaders
from src.config import (
    CLEAN_DATA_PKL, MODEL_PATH, VOCAB_PATH, TEST_SET_PATH, PLOT_PATH,
    WINDOW_SIZE, HIDDEN_SIZE, EMBEDDING_DIM, NUM_LAYERS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT, NUM_WORKERS
)

# Alias to match config.py variables
DATA_PATH = CLEAN_DATA_PKL

logger = setup_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    
scaler = GradScaler(enabled=(device.type == "cuda"))


def train():

    logger.info("Loading data...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, 'rb') as f:
        songs = pickle.load(f)
        
    # Load vocabulary
    logger.info("Loading vocabulary...")
    if not os.path.exists(VOCAB_PATH):
        logger.error(f"Vocabulary not found at {VOCAB_PATH}. Run src/model/vocabulary.py first to build it.")
        return

    vocab = Vocabulary()
    vocab.load_vocab(VOCAB_PATH)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create loaders for train 80% and test 20%
    logger.info("Creating dataloaders...")
    train_loader, test_loader = create_loaders(songs, vocab, BATCH_SIZE, WINDOW_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Save test set for evaluation later
    # logger.info(f"Saving test set to {TEST_SET_PATH}...")
    # all_X_test = []
    # all_y_test = []
    # for X, y in test_loader:
    #     all_X_test.append(X)
    #     all_y_test.append(y)
    
    # if all_X_test:
    #     X_test_tensor = torch.cat(all_X_test)
    #     y_test_tensor = torch.cat(all_y_test)
    #     with open(TEST_SET_PATH, 'wb') as f:
    #         pickle.dump({'X_test': X_test_tensor, 'y_test': y_test_tensor}, f)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    logger.info(f"Training on {device}")
    
    model = ChordLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,        # o leggermente più alto es. 2*LEARNING_RATE
    total_steps=EPOCHS * len(train_loader)
)

     
    best_val_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss_steps': []
    }
    
    global_step = 0
    running_loss = 0.0
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch_X, batch_y in pbar:
            global_step += 1
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
            # Backward pass with gradient scaling   
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step optimizer and scheduler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            running_loss += loss.item()
            
            # Save loss every 1000 steps
            if global_step % 1000 == 0:
                avg_running_loss = running_loss / 1000
                history['train_loss_steps'].append((global_step, avg_running_loss))
                running_loss = 0.0
            
            _, predicted = torch.max(outputs.data, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            current_loss = total_loss / (pbar.n + 1)
            current_acc = 100 * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
            
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"(*) Best Model Saved with Acc: {best_val_acc:.2f}%")
    
        logger.info(f"Saving training plots to {PLOT_PATH}...")
        plot_training_history(history, PLOT_PATH)
        
        history_path = PLOT_PATH.replace('.png', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        logger.info(f"Training history saved to {history_path}")        
            
            
    logger.info("Training complete.")
    logger.info(f"Best Test Accuracy: {best_val_acc:.2f}%")
    
    
    
def plot_training_history(history, save_path):
    """
    Plots and saves training/validation loss and accuracy
    
    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path where to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()    


if __name__ == "__main__":
    train()
