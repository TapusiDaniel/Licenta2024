import torch
import torch.nn as nn
from vgg19_model import VGG19
from bert_model import BertModel, BertConfig

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, bert_config, image_feature_size, dropout_prob=0.5):
        super(EarlyFusionModel, self).__init__()
        self.vgg19 = VGG19(num_classes)
        self.bert = BertModel(bert_config)

        self.vgg19_fc = nn.Linear(25088, bert_config.hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(bert_config.hidden_size * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_pooled_output = bert_outputs[1]
        
        vgg19_features = self.vgg19.features(image)
        vgg19_features = vgg19_features.view(vgg19_features.size(0), -1)
        
        vgg19_features = self.vgg19_fc(vgg19_features)
        
        fused_features = torch.cat((bert_pooled_output, vgg19_features), dim=1)
        output = self.fusion(fused_features)
        
        return output
    
class LateFusionModel(nn.Module):
    def __init__(self, num_classes, bert_config, dropout_prob=0.5):
        super(LateFusionModel, self).__init__()
        self.vgg19 = VGG19(num_classes)
        self.bert = BertModel(bert_config)
        
        self.text_classifier = nn.Linear(bert_config.hidden_size, num_classes)

        self.image_classifier = nn.Linear(25088, num_classes)
        
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_pooled_output = bert_outputs[1]
        
        vgg19_features = self.vgg19.features(image)
        vgg19_features = vgg19_features.view(vgg19_features.size(0), -1)
        
        text_output = self.text_classifier(bert_pooled_output)
        image_output = self.image_classifier(vgg19_features)
        
        fused_output = torch.cat((text_output, image_output), dim=1)
        output = self.fusion(fused_output)
        
        return output