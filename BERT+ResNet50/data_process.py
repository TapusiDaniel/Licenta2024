import os
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from config import Config
from sklearn.model_selection import train_test_split

def split_data(text_dirs, image_dirs, labels, test_size=0.2, random_state=42):
    text_files = []
    image_files = []
    labels_list = []

    for text_dir, image_dir, label in zip(text_dirs, image_dirs, labels):
        text_files.extend([os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith('.txt')])
        image_files.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        labels_list.extend([label] * len(os.listdir(text_dir)))

    train_text_files, test_text_files, train_image_files, test_image_files, train_labels, test_labels = train_test_split(
        text_files, image_files, labels_list, test_size=test_size, random_state=random_state, stratify=labels_list
    )

    return train_text_files, train_image_files, train_labels, test_text_files, test_image_files, test_labels

config = Config()

class FakeNewsDataset(Dataset):
    def __init__(self, text_files, image_files, labels, label_map, tokenizer, image_size=224):
        self.text_files = text_files
        self.image_files = image_files
        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer  
        self.image_size = image_size
        self.max_seq_length = 128
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data = list(zip(text_files, image_files, labels))
    
    def _load_data(self):
        data = []
        for label, (text_dir, image_dir) in enumerate(zip(self.text_dirs, self.image_dirs)):
            for filename in os.listdir(text_dir):
                if filename.endswith('.txt'):
                    text_path = os.path.join(text_dir, filename)
                    image_name = os.path.splitext(filename)[0]
                    image_extensions = ['.jpg', '.jpeg', '.png']
                    for ext in image_extensions:
                        image_path = os.path.join(image_dir, image_name + ext)
                        if os.path.exists(image_path):
                            data.append((text_path, image_path, self.labels[label]))
                            break
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_path, image_path, label = self.data[idx]
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        token_type_ids = torch.zeros_like(input_ids)  
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        label_idx = self.label_map[label]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 
            'image': image,
            'label': torch.tensor(label_idx, dtype=torch.long)
        }

def load_dataset(text_files, image_files, labels, label_map, tokenizer, image_size=224, batch_size=16, shuffle=True):
    dataset = FakeNewsDataset(text_files, image_files, labels, label_map, tokenizer, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader