# fpu/src/ai/AdvancedModels.py

import torch
from torchvision.models import resnet50, densenet121
from transformers import BertModel, BertTokenizer

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000):
        super(VisionTransformer, self).__init__()
        self.model = resnet50(pretrained=True)
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseNetClassifier, self).__init__()
        self.model = densenet121(pretrained=True)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class TextSentimentBert(nn.Module):
    def __init__(self, num_labels=2):
        super(TextSentimentBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class ReinforcementLearningAgent(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ReinforcementLearningAgent, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
