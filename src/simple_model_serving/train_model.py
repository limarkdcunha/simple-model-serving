from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load a pre-trained DistilBERT model for text classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Save model and tokenizer
model.save_pretrained("text_model")
print("✅ Model saved to 'text_model'!")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("text_model")
print("✅ Tokenizer saved to 'text_model'!")
