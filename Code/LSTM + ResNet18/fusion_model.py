import torch
import torch.nn as nn
from resnet18 import resnet18
from lstm_model import LSTMForSequenceClassification, LSTMConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, lstm_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.lstm = LSTMForSequenceClassification(lstm_config, num_classes)
        
        self.resnet_fc = nn.Linear(1000, lstm_config.hidden_size * 2 if lstm_config.bidirectional else lstm_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(lstm_config.hidden_size * 4 if lstm_config.bidirectional else lstm_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, lstm_pooled_output = self.lstm(input_ids, lengths, attention_mask)
        
        resnet_features = self.resnet(image)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        resnet_features = self.resnet_fc(resnet_features)

        fused_features = torch.cat((lstm_pooled_output, resnet_features), dim=1)
        output = self.fusion(fused_features)
        
        return output

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, lstm_config, image_feature_size, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.lstm = LSTMForSequenceClassification(lstm_config, num_classes)
        
        self.image_classifier = nn.Linear(1000, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(lstm_config.hidden_size * 2 + num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, text_output = self.lstm(input_ids, lengths, attention_mask)
        
        resnet_features = self.resnet(image)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        image_output = self.image_classifier(resnet_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output