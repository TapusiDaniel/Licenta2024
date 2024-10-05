import torch
import torch.nn as nn

class GRUConfig:
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_prob, bidirectional):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

class GRUModel(nn.Module):
    def __init__(self, config):
        super(GRUModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.gru = nn.GRU(config.embedding_size, config.hidden_size, config.num_layers, dropout=config.dropout_prob, bidirectional=config.bidirectional)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        self.pool = nn.AdaptiveAvgPool2d((1, config.hidden_size * 2 if config.bidirectional else config.hidden_size))
        
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
    
    def forward(self, input_ids, lengths, attention_mask):
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        pooled_output = self.pool(output.unsqueeze(1)).squeeze(1)
        return output, pooled_output

class GRUForSequenceClassification(nn.Module):
    def __init__(self, config, num_classes):
        super(GRUForSequenceClassification, self).__init__()
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        
        self.embedding = nn.Embedding(config.vocab_size, self.embedding_size)
        
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(config.dropout_prob)
        
        self.classifier = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, num_classes)

    def forward(self, input_ids, lengths, attention_mask):
        embeddings = self.embedding(input_ids)
        
        packed_sequence = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, hidden = self.gru(packed_sequence)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        output = self.dropout(output)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        pooled_output = self.dropout(hidden)
        
        return output, pooled_output