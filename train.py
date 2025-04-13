import numpy as np
from sklearn.utils import shuffle

def train(model, X_train, y_train, X_val, y_val, 
          epochs=100, batch_size=128, learning_rate=0.01, 
          reg_lambda=0.01, lr_decay=0.95, decay_every=10):
    
    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0
    best_params = None
    
    for epoch in range(epochs):
        # 学习率衰减
        if epoch % decay_every == 0 and epoch > 0:
            learning_rate *= lr_decay
        
        # 打乱数据
        X_shuffled, y_shuffled = shuffle(X_train, y_train)
        
        epoch_loss = 0.0
        for i in range(0, X_train.shape[0], batch_size):
            # 获取批次数据
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 前向传播
            probs = model.forward(X_batch)
            
            # 计算损失
            log_probs = -np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8)
            data_loss = np.mean(log_probs)
            reg_loss = 0.5 * reg_lambda * (np.sum(model.W1**2) + np.sum(model.W2**2))
            loss = data_loss + reg_loss
            epoch_loss += loss
            
            # 反向传播
            dW1, db1, dW2, db2 = model.backward(X_batch, y_batch, probs, reg_lambda)
            
            # 参数更新
            model.W1 -= learning_rate * dW1
            model.b1 -= learning_rate * db1
            model.W2 -= learning_rate * dW2
            model.b2 -= learning_rate * db2
        
        # 计算验证集指标
        val_probs = model.forward(X_val)
        val_loss = compute_loss(val_probs, y_val, model, reg_lambda)
        val_acc = compute_accuracy(val_probs, y_val)
        
        # 记录指标
        train_losses.append(epoch_loss / (X_train.shape[0]//batch_size))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = model.get_params()
        
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    # 恢复最佳参数
    model.set_params(best_params)
    return train_losses, val_losses, val_accs

def compute_loss(probs, y_true, model, reg_lambda):
    log_probs = -np.log(probs[np.arange(len(y_true)), y_true] + 1e-8)
    data_loss = np.mean(log_probs)
    reg_loss = 0.5 * reg_lambda * (np.sum(model.W1**2) + np.sum(model.W2**2))
    return data_loss + reg_loss

def compute_accuracy(probs, y_true):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_true)