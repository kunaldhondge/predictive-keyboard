import torch
import pickle
from nltk.tokenize import word_tokenize
from model import PredictiveKeyboard

# Load Vocabulary
with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)

# Load Model
model = PredictiveKeyboard(vocab_size)
model.load_state_dict(torch.load("keyboard_model.pth"))
model.eval()

def predict_next_word(text, sequence_length=4):
    tokens = word_tokenize(text.lower())
    # Pad if sequence is too short, or truncate if too long
    if len(tokens) < sequence_length - 1:
         # In a real app we might pad, but here let's valid input length
         pass # Simple version assumes sufficient context or handles it
    
    # Take the last sequence_length - 1 tokens
    input_tokens = tokens[-(sequence_length-1):]
    
    # Convert to indices
    input_indices = [word2idx.get(word, 0) for word in input_tokens] # 0 as unk if not found (simple fix)
    
    # Convert to tensor
    input_tensor = torch.tensor(input_indices).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        
    return idx2word[predicted_idx]


if __name__ == "__main__":
    print("--- Sherlock's Predictive Keyboard ---")
    print("Type your phrase below. Type 'quit' to exit.")

    # 'while True' creates a loop that keeps running until we say 'break'
    while True:
        # 'input' stops the program and waits for you to type something
        user_input = input("\nEnter your text: ")

        # Check if the user wants to stop
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            # We call our function to get the prediction
            # Note: I've updated the logic below to handle top-k suggestions
            tokens = word_tokenize(user_input.lower())
            
            # Check if we have at least 3 words for context
            if len(tokens) < 3:
                print("Suggestion: Please type at least 3 words for better results!")
                continue

            # This part gets the TOP 3 instead of just the TOP 1
            input_tokens = tokens[-3:]
            input_indices = [word2idx.get(word, 0) for word in input_tokens]
            input_tensor = torch.tensor(input_indices).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                # torch.topk finds the 'k' highest numbers in the output
                # This gives us the 3 most likely words
                probs = torch.softmax(output, dim=1)
                top_values, top_indices = torch.topk(probs, 3)
                
                # Convert the numbers back to words
                suggestions = [idx2word[idx.item()] for idx in top_indices[0]]
                
            print(f"Suggestions: {suggestions}")

        except Exception as e:
            print(f"Error: {e}")
