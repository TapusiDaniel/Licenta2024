import torch
from torch.utils.data import DataLoader
from data_process import load_dataset
from fusion_model import EarlyFusionModel, LateFusionModel
from config import Config
from gru_model import GRUConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

config = Config()

def evaluate(model, dataloader, device, vocab):
    model.eval()
    predictions = []
    true_labels = []

    unk_token_id = 0 

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            input_ids = []
            for text in texts:
                text_ids = []
                for token in text.split():
                    if token in vocab:
                        text_ids.append(vocab[token])
                    else:
                        text_ids.append(unk_token_id)
                input_ids.append(text_ids)

            lengths = [len(ids) for ids in input_ids]
            max_length = max(lengths)
            input_ids_padded = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
            input_ids_padded = torch.tensor(input_ids_padded, dtype=torch.long).to(device)
            lengths = torch.tensor(lengths, dtype=torch.long)
            attention_mask = (input_ids_padded != 0).float().to(device)

            outputs = model(input_ids_padded, lengths, attention_mask, images)
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

    vocab = checkpoint['vocab']

    if config.model_type == 'early_fusion':
        model = EarlyFusionModel(config.num_classes, GRUConfig(len(vocab), config.embedding_size, config.hidden_size, config.num_layers, config.dropout_prob, config.bidirectional), config.image_feature_size, config.dropout_prob)
    elif config.model_type == 'late_fusion':
        model = LateFusionModel(config.num_classes, GRUConfig(len(vocab), config.embedding_size, config.hidden_size, config.num_layers, config.dropout_prob, config.bidirectional), config.image_feature_size, config.dropout_prob)
    else:
        raise ValueError('Invalid model type')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_accuracy, test_precision, test_recall, test_f1, test_report = evaluate(model, test_dataloader, device, vocab)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1-score: {test_f1:.4f}')
    print(f'Test Classification Report:\n{test_report}')

if __name__ == '__main__':
    main()