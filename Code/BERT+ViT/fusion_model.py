import torch
import torch.nn as nn
from torchvision.models import resnet50
from vgg16_model import VGG16
from ViT_model import vit_base_patch16_224
from bert_model import BertModel, BertConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, bert_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.bert = BertModel(bert_config)
        
        vit_output_dim = self.vit.head.in_features
        self.vit_fc = nn.Linear(vit_output_dim, bert_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(bert_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_pooled_output = bert_outputs[1]
        
        vit_features = self.vit.forward_features(image)
        vit_features = vit_features[:, 0]
        vit_features = self.vit_fc(vit_features)
        
        fused_features = torch.cat((bert_pooled_output, vit_features), dim=1)
        output = self.fusion(fused_features)
        
        return output
    
class LateFusionModel(nn.Module):
    def __init__(self, num_classes, bert_config, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.bert = BertModel(bert_config)
        
        vit_output_dim = self.vit.head.in_features
        self.text_classifier = nn.Linear(bert_config.hidden_size, num_classes)
        self.image_classifier = nn.Linear(vit_output_dim, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_pooled_output = bert_outputs[1]
        
        vit_features = self.vit.forward_features(image)
        vit_features = vit_features[:, 0]  
        
        text_output = self.text_classifier(bert_pooled_output)
        image_output = self.image_classifier(vit_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output