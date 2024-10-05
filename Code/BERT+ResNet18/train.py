import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_process import load_dataset
from fusion_model import EarlyFusionModel, LateFusionModel
from config import Config
from data_process import split_data

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, token_type_ids, images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.bert_model)

    train_text_files, train_image_files, train_labels, test_text_files, test_image_files, test_labels = split_data(
        config.data_dirs, config.image_dirs, config.labels, test_size=0.2, random_state=42
    )

    train_dataloader = load_dataset(train_text_files, train_image_files, train_labels, config.label_map, tokenizer,
                                    batch_size=config.batch_size, shuffle=True)
    test_dataloader = load_dataset(test_text_files, test_image_files, test_labels, config.label_map, tokenizer,
                                   batch_size=config.batch_size, shuffle=False)
    
    if config.model_type == 'early_fusion':
        model = EarlyFusionModel(config.num_classes, config.bert_config, config.image_feature_size, config.dropout_prob)
    elif config.model_type == 'late_fusion':
        model = LateFusionModel(config.num_classes, config.bert_config, config.dropout_prob)
    else:
        raise ValueError('Invalid model type')

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    torch.save({
    'model_state_dict': model.state_dict(),
    'test_dataloader': test_dataloader
}, config.model_save_path)

if __name__ == '__main__':
    main()