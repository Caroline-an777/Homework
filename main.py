from utils import load_cifar10, preprocess_data, split_validation
from test import test
from search import hyperparameter_search
from visualize import plot_results, plot_learning_curves
import numpy as np

def main():
    # ==================== 数据加载 ====================
    print("Loading CIFAR-10...")
    X_train_full, y_train_full, X_test, y_test = load_cifar10()
    
    # ==================== 数据预处理 ====================
    print("\nPreprocessing data...")
    X_train, X_val, y_train, y_val = split_validation(X_train_full, y_train_full)
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    
    # ==================== 超参数搜索 ====================
    print("\nStarting hyperparameter search...")
    search_results, best_result = hyperparameter_search(X_train, y_train, X_val, y_val)
    
    # ==================== 结果可视化 ====================
    print("\nVisualizing results...")
    plot_results(search_results[:4])  # 显示前4组参数结果
    plot_learning_curves(
        best_result['train_losses'],
        best_result['val_losses'],
        best_result['val_accs']
    )
    
    # ==================== 最终测试 ====================
    print("\nTesting best model...")
    best_model = best_result['model']
    test_acc = test(best_model, X_test, y_test)
    
    # ==================== 保存结果 ====================
    print("\nSaving results...")
    np.savez('best_model.npz', 
             W1=best_model.W1, 
             b1=best_model.b1, 
             W2=best_model.W2, 
             b2=best_model.b2)
    print("Model saved to best_model.npz")

if __name__ == "__main__":
    main()