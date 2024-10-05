import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_process import load_dataset
from fusion_model import EarlyFusionModel, LateFusionModel
from config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, images)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    report = classification_report(true_labels, predictions, digits=4)

    return accuracy, precision, recall, f1, report

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

    test_accuracy, test_precision, test_recall, test_f1, test_report = evaluate(model, test_dataloader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1-score: {test_f1:.4f}')
    print(f'Test Classification Report:\n{test_report}')

if __name__ == '__main__':
    main()