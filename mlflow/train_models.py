import mlflow
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert_embeddings(texts: list) -> np.ndarray:
    """Generate BERT embeddings for text inputs"""
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    model.eval()
    
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embeddings as sentence representations
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    return np.concatenate(embeddings, axis=0)

def create_engagement_score(df: pd.DataFrame) -> pd.Series:
    """Create target engagement score"""
    return (
        df['Engagements'] + 
        2 * df['Shares'] + 
        3 * df['Bookmarks'] +
        0.5 * df['Profile visits']
    )

def train(data_path: str):
    with mlflow.start_run():
        # Load and prepare data
        df = pd.read_csv(data_path)
        df['CleanText'] = df['Post text'].str.replace(r'http\S+', '', regex=True)
        
        # Create target variable
        df['EngagementScore'] = create_engagement_score(df)
        
        # Generate BERT embeddings
        mlflow.log_param("bert_model", BERT_MODEL_NAME)
        text_embeddings = get_bert_embeddings(df['CleanText'].tolist())
        
        # Prepare numerical features
        numerical_features = df[['Impressions', 'Replies', 'Detail expands']]
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(numerical_features)
        
        # Combine features
        X = np.concatenate([text_embeddings, scaled_numerical], axis=1)
        y = df['EngagementScore']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Hybrid model pipeline
        model = Pipeline([
            ('xgb', XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8
            ))
        ])
        
        # Train and evaluate
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Log metrics
        metrics = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        }
        mlflow.log_metrics(metrics)
        
        # Log model and artifacts
        mlflow.xgboost.log_model(model.named_steps['xgb'], "xgboost_model")
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")
        
        # Save BERT embeddings for inference
        np.save("text_embeddings.npy", text_embeddings)
        mlflow.log_artifact("text_embeddings.npy")
        
        # Register model
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/xgboost_model",
            "XplodeContentAI"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()
    train(args.data_path)