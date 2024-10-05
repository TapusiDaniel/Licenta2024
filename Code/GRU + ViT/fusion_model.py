import torch
import torch.nn as nn
from ViT_model import vit_base_patch16_224
from gru_model import GRUForSequenceClassification, GRUConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, gru_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.gru = GRUForSequenceClassification(gru_config, num_classes)
        
        vit_output_dim = self.vit.head.in_features
        self.vit_fc = nn.Linear(vit_output_dim, gru_config.hidden_size * 2 if gru_config.bidirectional else gru_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(gru_config.hidden_size * 4 if gru_config.bidirectional else gru_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, gru_pooled_output = self.gru(input_ids, lengths, attention_mask)
        
        vit_features = self.vit.forward_features(image)
        vit_features = vit_features[:, 0]
        vit_features = self.vit_fc(vit_features)

        fused_features = torch.cat((gru_pooled_output, vit_features), dim=1)
        output = self.fusion(fused_features)
        
        return output

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, gru_config, image_feature_size, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.gru = GRUForSequenceClassification(gru_config, num_classes)
        
        vit_output_dim = self.vit.head.in_features
        self.text_classifier = nn.Linear(gru_config.hidden_size * 2 if gru_config.bidirectional else gru_config.hidden_size, num_classes)
        self.image_classifier = nn.Linear(vit_output_dim, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(gru_config.hidden_size * 2 + num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, lengths, attention_mask, image):
        _, text_output = self.gru(input_ids, lengths, attention_mask)
        
        vit_features = self.vit.forward_features(image)
        vit_features = vit_features[:, 0] 
        image_output = self.image_classifier(vit_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output