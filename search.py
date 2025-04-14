from model import ThreeLayerNN
from train import train
import itertools

def hyperparameter_search(X_train, y_train, X_val, y_val):
    param_grid = {
        'hidden_size': [128,256,512,1024],
        'learning_rate': [0.1,0.05,0.01,0.005,0.001],
        'reg_lambda': [0.01,0.005,0.001],
        'activation': ['relu','sigmoid']
    }
    
    best_acc = 0.0
    best_params = {}
    results = []
    
    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        
        print("\n" + "="*60)
        print(f"Training with: {params}")
        
        # 初始化模型
        model = ThreeLayerNN(
            input_size=3072,
            hidden_size=params['hidden_size'],
            output_size=10,
            activation=params['activation']
        )
        
        # 训练模型
        train_losses, val_losses, val_accs = train(
            model, X_train, y_train, X_val, y_val,
            epochs=50,  # 快速测试用较少的epoch
            learning_rate=params['learning_rate'],
            reg_lambda=params['reg_lambda']
        )
        
        # 记录结果
        current_acc = val_accs[-1]
        results.append({
            'params': params,
            'val_acc': current_acc,
            'model': model,
            'train_losses': train_losses,
            'val_losses':val_losses,
            'val_accs':val_accs
        })
        
        # 更新最佳参数
        if current_acc > best_acc:
            best_acc = current_acc
            best_params = results[-1]
        
        print(f"Current Best Acc: {best_acc:.4f}")
    
    return results, best_params