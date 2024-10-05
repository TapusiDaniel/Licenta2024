import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_process import load_dataset
from fusion_model import EarlyFusionModel, LateFusionModel
from config import Config
from gru_model import GRUConfig
from data_process import split_data
import numpy as np

config = Config()

def train(model, dataloader, criterion, optimizer, device, vocab):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for batch in dataloader:
        text = batch['text']
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        lengths = torch.tensor([len(t.split()) for t in text])

        input_ids = [np.array([vocab.get(word, len(vocab)) for word in t.split()]) for t in text]
        
        max_length = max(len(ids) for ids in input_ids)
        input_ids_padded = np.zeros((len(input_ids), max_length))
        for i, ids in enumerate(input_ids):
            input_ids_padded[i, :len(ids)] = ids
        
        input_ids_padded = torch.tensor(input_ids_padded, dtype=torch.long)
        
        attention_mask = torch.zeros_like(input_ids_padded)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1

        input_ids_padded = input_ids_padded.to(device)
        attention_mask = attention_mask.to(device)
        lengths = lengths.to(device)

        lengths = lengths.cpu()

        outputs = model(input_ids_padded, lengths, attention_mask, images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids_padded.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_text_files, train_image_files, train_labels, test_text_files, test_image_files, test_labels = split_data(
        config.data_dirs, config.image_dirs, config.labels, test_size=0.2, random_state=42
    )

    vocab = {}
    for text_file in train_text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)

    train_dataloader = load_dataset(train_text_files, train_image_files, train_labels, config.label_map, None,
                                    batch_size=config.batch_size, shuffle=True)
    test_dataloader = load_dataset(test_text_files, test_image_files, test_labels, config.label_map, None,
                                   batch_size=config.batch_size, shuffle=False)
    
    if config.model_type == 'early_fusion':
        model = EarlyFusionModel(config.num_classes, GRUConfig(len(vocab), config.embedding_size, config.hidden_size, config.num_layers, config.dropout_prob, config.bidirectional), config.image_feature_size, config.dropout_prob)
    elif config.model_type == 'late_fusion':
        model = LateFusionModel(config.num_classes, GRUConfig(len(vocab), config.embedding_size, config.hidden_size, config.num_layers, config.dropout_prob, config.bidirectional), config.dropout_prob)
    else:
        raise ValueError('Invalid model type')

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device, vocab)
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    torch.save({
    'model_state_dict': model.state_dict(),
    'test_dataloader': test_dataloader,
    'vocab': vocab  
}, config.model_save_path)

if __name__ == '__main__':
    main()