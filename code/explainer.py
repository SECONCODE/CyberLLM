from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values

input_text = "Supported versions that are affected are 12.1.1 and 12.2.8."  
model = AutoModelForSequenceClassification.from_pretrained('XXX/fine_29.bin', num_labels=51)
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs, output_attentions=True)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view