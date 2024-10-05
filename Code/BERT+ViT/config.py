import os
from transformers import BertConfig

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

        self.bert_model = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.bert_config = BertConfig.from_pretrained(self.bert_model)

        self.model_type = 'late_fusion'  # 'early_fusion' or 'late_fusion'
        self.num_classes = 5
        self.image_feature_size = 2048
        self.dropout_prob = 0.3

        self.max_seq_length = 128
        self.batch_size = 8
        self.num_epochs = 20
        self.learning_rate = 1e-5

        self.model_save_path = '/home/danyez87/Licenta/model_bertViTlatelr1e-5dropout0.3epochs20.pt'