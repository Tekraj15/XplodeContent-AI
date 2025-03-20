import xgboost as xgb
from sklearn.model_selection import train_test_split
from feature_engineering import FeatureEngineer
import joblib

def train_model():
    # Load config
    df = pd.read_parquet(config['paths']['processed_data'])
    
    # Feature Engineering
    fe = FeatureEngineer(config['model_params']['bert_model'])
    bert_features = fe.get_bert_embeddings(df['post_text'].tolist())
    tfidf_features = fe.get_tfidf_features(df['post_text'].tolist())
    
    # Combine features
    X = np.concatenate([
        bert_features,
        tfidf_features,
        df[['impressions', 'shares', 'replies']].values
    ], axis=1)
    
    # Target: Engagement score
    y = df['engagements'] + 2*df['shares']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train XGBoost
    model = xgb.XGBRegressor(**config['model_params']['xgboost'])
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, "models/xgboost/virality_model.pkl")
    
if __name__ == "__main__":
    train_model()