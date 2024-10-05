import torch
import torch.nn as nn
from resnet34 import resnet34
from gru_model import GRUForSequenceClassification, GRUConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, gru_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.gru = GRUForSequenceClassification(gru_config, num_classes)
        
        self.resnet_fc = nn.Linear(1000, gru_config.hidden_size * 2 if gru_config.bidirectional else gru_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(gru_config.hidden_size * 4 if gru_config.bidirectional else gru_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, gru_pooled_output = self.gru(input_ids, lengths, attention_mask)
        
        resnet_features = self.resnet(image)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        resnet_features = self.resnet_fc(resnet_features)

        fused_features = torch.cat((gru_pooled_output, resnet_features), dim=1)
        output = self.fusion(fused_features)
        
        return output

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, gru_config, image_feature_size, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.gru = GRUForSequenceClassification(gru_config, num_classes)
        
        self.text_classifier = nn.Linear(gru_config.hidden_size * 2 if gru_config.bidirectional else gru_config.hidden_size, num_classes)
        self.image_classifier = nn.Linear(1000, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(gru_config.hidden_size * 2 + num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, text_output = self.gru(input_ids, lengths, attention_mask)
        
        resnet_features = self.resnet(image)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        image_output = self.image_classifier(resnet_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output