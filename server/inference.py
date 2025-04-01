import torch
import torch.nn as nn
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Define the model architecture (must match training)
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model_and_components():
    """Load all necessary components for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load('sentiment_classifier_full.pth', map_location=device)
    
    # Initialize model
    model = SentimentClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load other components
    vectorizer = joblib.load('count_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    return model, vectorizer, label_encoder, device

def predict_sentiment(comment, model, vectorizer, label_encoder, device='cpu'):
    """
    Predict sentiment of a single comment.
    Returns:
        dict: {'sentiment': 'positive/neutral/negative', 'probabilities': dict}
    """
    # Preprocess the input
    with torch.no_grad():
        # Vectorize the comment
        X = vectorizer.transform([comment]).toarray()
        X = torch.tensor(X, dtype=torch.float32).to(device)
        
        # Make prediction
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        
        # Convert to human-readable labels
        sentiment = label_encoder.inverse_transform([predicted_class])[0]
        
        # Create probability dictionary
        class_probabilities = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        result = max(class_probabilities, key=class_probabilities.get)
        
        return result

# Example usage
if __name__ == "__main__":
    # Load everything
    model, vectorizer, label_encoder, device = load_model_and_components()
    
    # Example comments
    comments = [
        "I love this video! It's amazing!",
        "This is just okay, nothing special",
        "Terrible content, would not recommend",
        "The quality could be better but I like the concept"
        "I hate this content, it is worst ever"
    ]
    
    # Make predictions
    for comment in comments:
        result = predict_sentiment(comment, model, vectorizer, label_encoder, device)
        print(f"Comment: {comment}")
        print(f"Predicted sentiment: {result['sentiment']}")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 50)