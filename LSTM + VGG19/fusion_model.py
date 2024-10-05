import torch
import torch.nn as nn
from vgg19_model import VGG19
from lstm_model import LSTMForSequenceClassification, LSTMConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, lstm_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vgg19 = VGG19(num_classes)
        self.lstm = LSTMForSequenceClassification(lstm_config, num_classes)
        
        self.vgg19_fc = nn.Linear(25088, lstm_config.hidden_size * 2 if lstm_config.bidirectional else lstm_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(lstm_config.hidden_size * 4 if lstm_config.bidirectional else lstm_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, lstm_pooled_output = self.lstm(input_ids, lengths, attention_mask)
        
        vgg19_features = self.vgg19.features(image)
        vgg19_features = vgg19_features.view(vgg19_features.size(0), -1)
        
        vgg19_features = self.vgg19_fc(vgg19_features)
        
        fused_features = torch.cat((lstm_pooled_output, vgg19_features), dim=1)
        output = self.fusion(fused_features)
        
        return output

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, lstm_config, image_feature_size, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vgg19 = VGG19(num_classes)
        self.lstm = LSTMForSequenceClassification(lstm_config, num_classes)
        
        self.image_classifier = nn.Linear(25088, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(lstm_config.hidden_size * 2 + num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, text_output = self.lstm(input_ids, lengths, attention_mask)
        
        vgg19_features = self.vgg19.features(image)
        vgg19_features = vgg19_features.view(vgg19_features.size(0), -1)
        
        image_output = self.image_classifier(vgg19_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output