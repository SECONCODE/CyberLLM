
import os
import pickle
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader, TensorDataset, Dataset

'''
data = pd.read_csv('XXX//Updated ENISA EXTRACTED2.csv')
cve_ids = data['cve_id'].tolist()
descriptions = data['description'].tolist()
labels_str = data['label'].astype(str).tolist()

data = pd.read_excel('XXX//augmented_data2.xlsx')
cve_ids = data['cve_id'].tolist()
descriptions = data['description'].tolist()
techniques_str = data['technique_id'].astype(str).tolist()


combined_text = [f'{cve_id} {description}' for cve_id, description in zip(cve_ids, descriptions)]
mlb = MultiLabelBinarizer()
labels = [label.split(',') for label in labels_str]

for i1 in range(len(labels)):
  for i2 in range(len(labels[i1])):
    labels[i1][i2]=labels[i1][i2].strip()

labels_encoded = mlb.fit_transform(labels)

class_maplie=mlb.classes_
class_maplie=["'1007'", "'1012'", "'1014'", "'1016'", "'1018'", "'1027'", "'1033'", "'1037'", "'1039'", "'1046'", "'1049'", "'1057'", "'1062'", "'1069'", "'1070.005'", "'1080'", "'1082'", "'1083'", "'1087'", "'1090'", "'1110'", "'1120'", "'1124'", "'1134'", "'1135'", "'1185'", "'1201'", "'1505.003'", "'1542.003'", "'1543.001'", "'1543.003'", "'1543.004'", "'1546.001'", "'1546.004'", "'1546.008'", "'1547.006'", "'1547.008'", "'1547.009'", "'1552.001'", "'1552.002'", "'1553.004'", "'1558.003'", "'1562.001'", "'1562.003'", "'1569.001'", "'1574.001'", "'1574.004'", "'1574.010'", "'1574.011'", "'1647'", 'nan']

if os.path.exists('XXX//X_train.pkl'):
    X_train = pickle.load(open('XXX//X_train.pkl','rb'))
    X_test = pickle.load(open('XXX//X_test.pkl','rb'))
    y_train = pickle.load(open('XXX//y_train.pkl','rb'))
    y_test = pickle.load(open('XXX//y_test.pkl','rb'))
else:
    X_train, X_test, y_train, y_test = train_test_split(combined_text, labels_encoded, test_size=0.2)
    pickle.dump(X_train, open('XXX//X_train.pkl','wb'))
    pickle.dump(X_test, open('XXX//X_test.pkl','wb'))
    pickle.dump(y_train, open('XXX//y_train.pkl','wb'))
    pickle.dump(y_test, open('XXX//y_test.pkl','wb'))
'''

lu='XXX'

train_aug = True

if train_aug:
    X_train = pickle.load(open(lu+'dataset/X_train_aug.pkl','rb'))
    X_test = pickle.load(open(lu+'dataset/X_test.pkl','rb'))
    y_train = pickle.load(open(lu+'dataset/y_train_aug.pkl','rb'))
    y_test = pickle.load(open(lu+'dataset/y_test.pkl','rb'))
else:
    X_train = pickle.load(open(lu+'dataset/X_train.pkl','rb'))
    X_test = pickle.load(open(lu+'dataset/X_test.pkl','rb'))
    y_train = pickle.load(open(lu+'dataset/y_train.pkl','rb'))
    y_test = pickle.load(open(lu+'dataset/y_test.pkl','rb'))


model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=51)
#model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
tokenizer.pad_token = tokenizer.eos_token

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024): #128
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        encoding['labels'] = torch.tensor(label, dtype=torch.float32)
        return encoding


train_dataset = CustomDataset(X_train, y_train, tokenizer)
test_dataset = CustomDataset(X_test, y_test, tokenizer)


batch_size = 1
epochs = 3
learning_rate = 2e-5
num_labels=51

tokenizer.save_pretrained(lu+'save/token.json')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
aaa=model.train()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(30):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids.squeeze(dim=1), attention_mask=attention_mask.squeeze(dim=1))#
        logits = outputs.logits
        #average_logits = torch.mean(logits, dim=1)
        #probabilities = torch.sigmoid(average_logits)
        #loss = F.binary_cross_entropy_with_logits(probabilities, labels)
        loss = criterion(logits.view(-1, 51), labels.view(-1, 51))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}")
    model.save_pretrained(lu+'save/fine_'+str(epoch)+'.bin')



'''
aaa=model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_dataloader:#test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids.squeeze(dim=1), attention_mask=attention_mask.squeeze(dim=1))
        logits = outputs.logits
        #logits = torch.mean(logits, dim=1)
        predictions = torch.sigmoid(logits)
        predictions[predictions>0.5]=1
        predictions[predictions<=0.5]=0
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='micro')
recall = recall_score(all_labels, all_preds, average='micro')
f1 = f1_score(all_labels, all_preds, average='micro')
'''




