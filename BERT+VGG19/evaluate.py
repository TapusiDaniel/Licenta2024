import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_process import load_dataset
from fusion_model import EarlyFusionModel, LateFusionModel
from config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, dataloader, device, config):
    model.eval()
    predictions = []
    true_labels = []
    filenames = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            filenames.extend(batch['filename'])

            outputs = model(input_ids, attention_mask, token_type_ids, images)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    report = classification_report(true_labels, predictions, digits=4)
    cm = confusion_matrix(true_labels, predictions)

    examples_by_label = {label: [] for label in config.labels}

    for filename, true_label, pred_label in zip(filenames, true_labels, predictions):
        true_label_name = config.labels[true_label]
        pred_label_name = config.labels[pred_label]
        examples_by_label[true_label_name].append((filename, true_label_name, pred_label_name))

    for label in config.labels:
        print(f"{label}:")
        for example in examples_by_label[label]:
            filename, true_label_name, pred_label_name = example
            print(f"  Fișier: {filename}, Etichetă adevărată: {true_label_name}, Etichetă prezisă: {pred_label_name}")
        print()

    return accuracy, precision, recall, f1, report, cm

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(config.model_save_path)
    test_dataloader = checkpoint['test_dataloader']

    if config.model_type == 'early_fusion':
        model = EarlyFusionModel(config.num_classes, config.bert_config, config.image_feature_size, config.dropout_prob)
    elif config.model_type == 'late_fusion':
        model = LateFusionModel(config.num_classes, config.bert_config, config.dropout_prob)
    else:
        raise ValueError('Invalid model type')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_accuracy, test_precision, test_recall, test_f1, test_report, test_cm = evaluate(model, test_dataloader, device, config)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1-score: {test_f1:.4f}')
    print(f'Test Classification Report:\n{test_report}')
    print(f'Test Confusion Matrix:\n{test_cm}')

    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matricea de confuzie')
    plt.xlabel('Etichetele prezise')
    plt.ylabel('Etichetele adevărate')
    plt.tight_layout()
    plt.savefig('confusion_matrix_seaborn.png')

if __name__ == '__main__':
    main()