import torch
import torch.nn as nn
from vgg19_model import VGG19
from distilbert_model import DistilBertForSequenceClassification

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, distilbert_config, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vgg16 = VGG19(num_classes)
        self.distilbert = DistilBertForSequenceClassification(distilbert_config, num_classes)
        
        self.vgg16_fc = nn.Linear(25088, distilbert_config.hidden_size)
        
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
        
        vgg16_features = self.vgg16.features(image)
        vgg16_features = vgg16_features.view(vgg16_features.size(0), -1)
        vgg16_features = self.vgg16_fc(vgg16_features)
        
        distilbert_pooled_output = distilbert_pooled_output.unsqueeze(1).expand(-1, vgg16_features.size(1))
        
        fused_features = torch.cat((distilbert_pooled_output, vgg16_features), dim=1)
        
        output = self.fusion(fused_features)
        
        return output
        
class LateFusionModel(nn.Module):
    def __init__(self, num_classes, distilbert_config, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vgg16 = VGG19(num_classes)
        self.distilbert = DistilBertForSequenceClassification(distilbert_config, num_classes)
        
        self.image_classifier = nn.Linear(25088, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        distilbert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        text_output = distilbert_outputs[0]

        vgg16_features = self.vgg16.features(image)
        vgg16_features = vgg16_features.view(vgg16_features.size(0), -1)
        
        image_output = self.image_classifier(vgg16_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output