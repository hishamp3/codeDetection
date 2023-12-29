from django.db import models
import os
import gdown
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification
from torch import nn
import torch

model_name = 'microsoft/codebert-base'
FILE = 'CodeBERT.pth'


class Classifier(nn.Module):

    def __init__(self, dropout=0.10):
        super(Classifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 5)

    def forward(self, input_id, mask):
        _, pooled_output = self.model(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)

        # First Layer
        linear_output = self.linear1(dropout_output)
        layer_output = self.relu(linear_output)

        # Second Layer
        linear_output = self.linear2(layer_output)
        final_layer = self.relu(linear_output)

        return final_layer


class LLM(models.Model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    filepath = os.path.join(os.getcwd(), FILE)

    if os.path.exists(filepath):
        pass
    else:
        gdown.download(id='1IlqNQiNhH8FLs2GVKy0mN0JgCXnZPCT3', output=filepath, quiet=False)

    model = Classifier()
    model.load_state_dict(torch.load(filepath))
    model.eval()

    def tokenize(self, code):
        code_token = self.tokenizer(code, padding='max_length', max_length=512, truncation=True,
                                    return_tensors="pt").to(self.device)
        return code_token

    def detection(self, token) -> str:
        output = self.model(token['input_ids'], token['attention_mask'])
        logit = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        prediction = "Defective" if logit[0] == 0 else "Not Defective"
        return prediction
