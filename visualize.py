import matplotlib.pyplot as plt

def plot_results(results):
    for result in results:
        plt.figure(figsize=(12, 5))  # 宽幅适合并排显示
        
        params = result['params']
        
        # ================= 生成文件名 =================
        # 构造参数标识字符串（示例：hs128_lr0.01_reg0.1_relu）
        param_str = "_".join([
            f"hs{params['hidden_size']}",
            f"lr{params['learning_rate']:.0e}".replace(".", "_"),  
            f"reg{params['reg_lambda']:.0e}",
            f"act{params['activation']}"
        ])
        filename = f"result_{param_str}.png"
        
        # ================= 损失子图 =================
        plt.subplot(1, 2, 1)  # 1行2列，第1个位置
        plt.plot(result['train_losses'], label='Train Loss')
        plt.plot(result['val_losses'], label='Val Loss', linestyle='--')
        plt.title("Loss Curves")  # 简化标题
        plt.xlabel('Epochs')
        plt.legend()
        
        # ================= 准确率子图 =================
        plt.subplot(1, 2, 2)  # 1行2列，第2个位置
        plt.plot(result['val_accs'], color='green', label='Val Accuracy')
        plt.title("Accuracy Curve")  # 简化标题
        plt.xlabel('Epochs')
        plt.legend()

        # ================= 保存 & 清理 =================
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭当前图表，避免内存泄漏

def plot_learning_curves(train_losses, val_losses, val_accs):
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, color='green', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()