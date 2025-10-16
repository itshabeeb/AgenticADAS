"""
Intent Classification Module using DistilBERT.
"""
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from typing import Tuple, Dict

class IntentClassifier:
    def __init__(self, model_path: str):
        """
        Initialize the intent classifier with a fine-tuned DistilBERT model.
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.intent_labels = ["Normal", "Priority", "Critical", "Cautious", "Eco"]
        
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            tuple: (intent_label, confidence_score)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities).item()
            
        intent_label = self.intent_labels[prediction.item()]
        return intent_label, confidence
    
    def get_intent_data(self, text: str) -> Dict:
        """
        Get structured intent data including label and confidence.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Intent classification results
        """
        intent_label, confidence = self.classify_intent(text)
        return {
            "text": text,
            "intent": intent_label,
            "confidence": confidence
        }
