from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

class FeatureEngineer:
    def __init__(self, bert_model_name: str):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.tfidf = TfidfVectorizer(max_features=2000)
        
    def get_bert_embeddings(self, texts: list) -> np.array:
        inputs = self.bert_tokenizer(texts, return_tensors='pt', 
                                   padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def get_tfidf_features(self, texts: list) -> np.array:
        return self.tfidf.fit_transform(texts).toarray()