import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, classes):
    # Rilevamento automatico del device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Disabilitiamo il calcolo dei gradienti per velocizzare l'inferenza
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Stampa del Report (Accuracy, Precision, Recall, F1-Score)
    print("\n--- Report di Classificazione sul Test Set ---")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Disegno della Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Etichetta Reale')
    plt.xlabel('Predizione del Modello')
    plt.title('Confusion Matrix Multiclasse')
    plt.show() # <-- Aggiunto per mostrare il grafico correttamente

def plot_roc_curve_multiclass(model, test_loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Estraiamo le probabilità usando il Softmax
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Binarizziamo le etichette per l'approccio One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    n_classes = y_true_bin.shape[1]
    
    # Calcoliamo ROC e AUC per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'ROC curve di {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve Multiclasse (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()

 