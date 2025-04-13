import numpy as np
from sklearn.metrics import classification_report

def test(model, X_test, y_test):
    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test)
    
    print("="*60)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=[str(i) for i in range(10)]))
    print("="*60)
    
    return accuracy