import torch
import torch.nn as nn
from ViT_model import vit_base_patch16_224
from distilbert_model import DistilBertForSequenceClassification

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, distilbert_config, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.distilbert = DistilBertForSequenceClassification(distilbert_config, num_classes)
        
        vit_output_dim = self.vit.head.in_features
        self.vit_fc  = nn.Linear(vit_output_dim, distilbert_config.hidden_size)
        
        fusion_input_size = 2 * distilbert_config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        distilbert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        distilbert_pooled_output = distilbert_outputs[0][:, 0]
        
        vit_features = self.vit.forward_features(image)
        vit_features = vit_features[:, 0]
        vit_features = self.vit_fc(vit_features)
        
        distilbert_pooled_output = distilbert_pooled_output.unsqueeze(1).expand(-1, vit_features.size(1))
        
        fused_features = torch.cat((distilbert_pooled_output, vit_features), dim=1)
        
        output = self.fusion(fused_features)
        
        return output
        
class LateFusionModel(nn.Module):
    def __init__(self, num_classes, distilbert_config, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.distilbert = DistilBertForSequenceClassification(distilbert_config, num_classes)

        vit_output_dim = self.vit.head.in_features
        self.image_classifier = nn.Linear(vit_output_dim, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        distilbert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        text_output = distilbert_outputs[0]

        vit_features = self.vit.forward_features(image)
        vit_features = vit_features[:, 0] 
        
        image_output = self.image_classifier(vit_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output