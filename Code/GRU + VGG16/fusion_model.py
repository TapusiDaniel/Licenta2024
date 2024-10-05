import torch
import torch.nn as nn
from vgg16_model import VGG16
from gru_model import GRUForSequenceClassification, GRUConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, gru_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vgg16 = VGG16(num_classes)
        self.gru = GRUForSequenceClassification(gru_config, num_classes)
        
        self.vgg16_fc = nn.Linear(25088, gru_config.hidden_size * 2 if gru_config.bidirectional else gru_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(gru_config.hidden_size * 4 if gru_config.bidirectional else gru_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, gru_pooled_output = self.gru(input_ids, lengths, attention_mask)
        
        vgg16_features = self.vgg16.features(image)
        vgg16_features = vgg16_features.view(vgg16_features.size(0), -1)
        vgg16_features = self.vgg16_fc(vgg16_features)

        fused_features = torch.cat((gru_pooled_output, vgg16_features), dim=1)
        output = self.fusion(fused_features)
        
        return output

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, gru_config, image_feature_size, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vgg16 = VGG16(num_classes)
        self.gru = GRUForSequenceClassification(gru_config, num_classes)
        
        self.text_classifier = nn.Linear(gru_config.hidden_size * 2 if gru_config.bidirectional else gru_config.hidden_size, num_classes)
        self.image_classifier = nn.Linear(25088, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(gru_config.hidden_size * 2 + num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, text_output = self.gru(input_ids, lengths, attention_mask)
        
        vgg16_features = self.vgg16.features(image)
        vgg16_features = vgg16_features.view(vgg16_features.size(0), -1)
        image_output = self.image_classifier(vgg16_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output