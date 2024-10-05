import os
import numpy as np

class Config:
    def __init__(self):
        self.data_dir = '/home/danyez87/Licenta/Fakerom_final'
        
        self.data_dirs = [
            os.path.join(self.data_dir, 'știri fabricate'),
            os.path.join(self.data_dir, 'știri propagandistice'),
            os.path.join(self.data_dir, 'știri reale'),
            os.path.join(self.data_dir, 'știri plauzibile'),
            os.path.join(self.data_dir, 'știri satirice')
        ]
        
        self.image_dirs = [
            os.path.join(self.data_dir, 'știri fabricate_img'),
            os.path.join(self.data_dir, 'știri propagandistice_img'),
            os.path.join(self.data_dir, 'știri reale_img'),
            os.path.join(self.data_dir, 'știri plauzibile_img'),
            os.path.join(self.data_dir, 'știri satirice_img')
        ]
        
        self.labels = ['fabricate', 'propagandistice', 'reale', 'plauzibile', 'satirice']
        
        self.label_map = {
            'fabricate': 0,
            'propagandistice': 1,
            'reale': 2,
            'plauzibile': 3,
            'satirice': 4
        }
        
        self.model_type = 'early_fusion'  # 'early_fusion' or 'late_fusion'
        self.num_classes = 5
        self.image_feature_size = 1000
        self.dropout_prob = 0.3
        
        self.max_seq_length = 128
        self.batch_size = 8
        self.num_epochs = 20
        self.learning_rate = 1e-5
        
        self.model_save_path = '/home/danyez87/Licenta/model_bigruresnet101earlylr1e-5dropout0.3epochs20batch8.pt'
        
        self.embedding_size = 300
        self.hidden_size = 128
        self.num_layers = 2
        self.bidirectional = True
        
        self.embedding_file = '/home/danyez87/Licenta/nlpl/model.txt'
        self.word_embeddings, self.vocab_size = self.load_word_embeddings()
    
    def load_word_embeddings(self):
        word_embeddings = {}
        with open(self.embedding_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = vector
        
        vocab_size = len(word_embeddings)
        return word_embeddings, vocab_size