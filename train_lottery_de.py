import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from data_reading import read_xoso

num_day_steps = 30
vocab_size = 100

class SpecialPrizeDataset(Dataset):
    def __init__(self, data, num_day_steps, vocab_size):
        self.data = data
        self.num_day_steps = num_day_steps
        self.vocab_size = vocab_size

        self.X = []
        self.y = []

        self._prepare_data()

    def _prepare_data(self):
        for i in range(len(self.data) - self.num_day_steps):
            input_sequence = self.data[i : i + self.num_day_steps]
            self.X.append(torch.tensor(input_sequence, dtype=torch.long))

            target_number = self.data[i+self.num_day_steps]
            self.y.append(torch.tensor(target_number, dtype=torch.long))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
class SpecialPrizeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout_rate=0.2):
        super(SpecialPrizeModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.pos_encoding = nn.Parameter(torch.randn(500, embedding_dim) * 0.1)

        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)

        if seq_len <= self.pos_encoding.size(0):
            pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)
            embedded = embedded + pos_enc

        embedded = self.embedding_dropout(embedded)

        lstm_ouput, (h_n, c_n) = self.lstm(embedded)

        # final_hidden = h_n[-1]

        attn_output, attn_weights = self.attention(lstm_ouput, lstm_ouput, lstm_ouput)

        lstm_ouput = self.layer_norm1(lstm_ouput + attn_output)

        final_output = lstm_ouput[:, -1, :]

        x1 = self.dropout(self.gelu(self.fc1(final_output)))
        x1 = self.layer_norm2(x1 + final_output)  # Residual connection
        
        x2 = self.dropout(self.gelu(self.fc2(x1)))
        output = self.fc3(x2)

        return output
    
def predict_special_prize(model, data, device, topk=10):
    model.eval()
    input_sequence = data[-num_day_steps:]
    input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities[0], topk)

        idx_and_probs = dict()
        for i in range(topk):
            idx_and_probs[top_indices[i].item()] = top_probs[i].item()
                
        return idx_and_probs


def train_predict_de_bac(epoch=45):
    special_data = read_xoso("xs_data/xsmb_data.json", "de", "bac")
    special_dataset = SpecialPrizeDataset(
        data=special_data,
        num_day_steps=num_day_steps,
        vocab_size=vocab_size
    )

    train_size = int(0.8 * len(special_dataset))
    val_size = int(0.1 * len(special_dataset))
    test_size = len(special_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        special_dataset, [train_size, val_size, test_size]
    )

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Tổng số mẫu trong dataset: {len(special_dataset)}")
    print(f"Số mẫu huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu validation: {len(val_dataset)}")
    print(f"Số mẫu kiểm tra: {len(test_dataset)}")

    # Kiểm tra một batch
    for batch_X, batch_y in train_loader:
        print(f"\nInput batch X shape: {batch_X.shape}")  # (batch_size, num_day_steps)
        print(f"Output batch y shape: {batch_y.shape}")   # (batch_size,)
        print(f"Ví dụ input sequence: {batch_X[0]}")
        print(f"Ví dụ target: {batch_y[0].item()}")
        break

    embedding_dim = 32
    hidden_size = 128
    num_layers = 2
    dropout_rate = 0.3

    model = SpecialPrizeModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    num_epochs = epoch
    best_val_loss = float('inf')

    print("Bắt đầu training...")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Quá trình training
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device),  targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        # Quá trình val
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), f'model/special_lottery_model_{num_epochs}.pth')
    print("Training completed!")

    model.load_state_dict(torch.load(f'model/special_lottery_model_{num_epochs}.pth'))

    # Giai đoạn test
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_accuracy = test_correct / test_total
    print(f"\nTest Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    nums_and_probs = predict_special_prize(model, special_dataset.data, device)
    return nums_and_probs

if __name__ == "__main__":
    nums_and_probs = train_predict_de_bac()
    for num, prob in nums_and_probs.items():
        print(f"{num}: {prob:.2f}")
