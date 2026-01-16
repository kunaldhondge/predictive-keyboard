import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import Counter
from model import PredictiveKeyboard

# 1. Load and Tokenize
with open('sherlock.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
tokens = word_tokenize(text)

# 2. Build Vocabulary
word_counts = Counter(tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

# 3. Create Sequences
sequence_length = 4
data = []
for i in range(len(tokens) - sequence_length):
    input_seq = [word2idx[word] for word in tokens[i:i + sequence_length - 1]]
    target = word2idx[tokens[i + sequence_length - 1]]
    data.append((torch.tensor(input_seq), torch.tensor(target)))

# 4. Training Loop
print("Starting training...")
model = PredictiveKeyboard(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

for epoch in range(10): # Adjust epochs as needed
    for input_seq, target in data[:5000]: # Small slice for testing
        output = model(input_seq.unsqueeze(0))
        loss = criterion(output, target.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")

# Save the model
# Save the model
torch.save(model.state_dict(), "keyboard_model.pth")

# Save the vocabulary
import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(word2idx, f)
print("Model and vocabulary saved.")