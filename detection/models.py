from django.db import models
import os
import torch
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification


class LLM(models.Model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(os.getcwd() + "\\CodeBERT")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def tokenize(self, code):
        code_token = self.tokenizer(code, padding='max_length', max_length=512, truncation=True,
                                    return_tensors="pt").to(self.device)
        return code_token

    def detection(self, token) -> str:
        output = self.model(token['input_ids'], token['attention_mask'])
        logit = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        prediction = "vulnerability" if logit[0] == 0 else "Not vulnerability"
        return prediction
