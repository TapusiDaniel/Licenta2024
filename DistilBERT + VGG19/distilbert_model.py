import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertModel, DistilBertConfig

class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(DistilBertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs