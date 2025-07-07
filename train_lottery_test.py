import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from data_reading import read_xoso

torch.manual_seed(42)
num_day_steps = 10  # Dự đoán dựa trên 10 ngày gần nhất
max_prizes_per_day = 68 # Số lượng số trúng thưởng
vocab_size = 100    # Số lượng số khả thi: 0 - 99

class LotteryDataset(Dataset):
    def __init__(self, data, num_day_steps, max_prizes_per_day, vocab_size):
        self.data = data
        self.num_day_steps = num_day_steps
        self.max_prizes_per_day = max_prizes_per_day
        self.vocab_size = vocab_size

        self.X = []
        self.lengths = []
        self.y = []

        self._prepared_data()

    def _prepared_data(self):
        for i in range(len(self.data) - self.num_day_steps):
            sequence = self.data[i : i + self.num_day_steps]
            padded_sequence = []
            length_sequence = []

            for day in sequence:
                actual_len = len(day)
                length_sequence.append(actual_len)
                padded = day + [vocab_size] * (self.max_prizes_per_day - actual_len)
                padded_sequence.append(torch.tensor(padded, dtype=torch.long))

            self.X.append(torch.stack(padded_sequence))
            self.lengths.append(torch.tensor(length_sequence, dtype=torch.long))

            target_next_day = self.data[i + self.num_day_steps]
            target_multi_hot = torch.zeros(self.vocab_size, dtype=torch.float32)
            for n in target_next_day:
                target_multi_hot[n] = 1.0
            self.y.append(target_multi_hot)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]  # Shape: (num_day_steps, max_prizes_per_day)
        y = self.y[index]  # Multi-hot vector
        lengths = (x != vocab_size).sum(dim=1)  # Số lượng số thực tế mỗi ngày
        return x, lengths, y

    
class LotteryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_prizes_per_day, hidden_size, num_layers=1, dropout_rate=0.0):
        super(LotteryModel, self).__init__()

        self.max_prizes_per_day = max_prizes_per_day
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_num = vocab_size

        self.embedding = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x, lengths):
        batch_size, num_day_steps, max_prizes_per_day = x.size()

        # Flatten theo ngày
        x_flatten = x.view(batch_size*num_day_steps, max_prizes_per_day)
        lengths_flat = lengths.view(batch_size * num_day_steps)

        # Nhúng số trúng thưởng
        embedded = self.embedding(x_flatten)
        
        # Tạo mask để loại bỏ padding
        mask = (x_flatten != self.padding_num).unsqueeze(-1)

        # Loại bỏ embedding của padding
        masked_embed = embedded * mask.float()

        # Trung bình embedding theo số lượng thực
        sum_embed = masked_embed.sum(dim=1)
        lengths_clamped = lengths_flat.clamp(min=1).unsqueeze(1).to(embedded.dtype)
        avg_embed = sum_embed / lengths_clamped

        daily_embed = avg_embed.view(batch_size, num_day_steps, self.embedding_dim)

        # LSTM → Attention → Residual
        lstm_output, _ = self.lstm(daily_embed)  # (B, T, H)
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        combined_output = lstm_output + attn_output

        # Lấy bước cuối cùng trong chuỗi thời gian
        final_output = combined_output[:, -1, :]  # (B, H)

        # FC layers với residual + dropout + norm
        x1 = self.dropout(self.gelu(self.fc1(final_output)))
        x1 = self.layer_norm1(x1 + final_output)

        x2 = self.dropout(self.gelu(self.fc2(x1)))
        x2 = self.layer_norm2(x2)

        output = self.fc3(x2)  # (B, vocab_size)
        return output
    
def predict_next_day(model, last_days_data, device):
    model.eval()

    # Lấy 10 ngày gần nhất
    recent_days = last_days_data[-num_day_steps:]  # list dài 10 phần tử, mỗi phần tử là list của 68 số

    # Chuẩn hoá: đảm bảo có đúng 68 số mỗi ngày và đã được padding bằng 100 nếu thiếu
    padded_input = []
    lengths = []
    for day in recent_days:
        actual_numbers = [num for num in day if num != vocab_size]
        lengths.append(len(actual_numbers))
        padded_day = day + [vocab_size] * (max_prizes_per_day - len(day))
        padded_input.append(padded_day[:max_prizes_per_day])  # đảm bảo độ dài đúng 68

    input_tensor = torch.tensor([padded_input], dtype=torch.long).to(device)        # shape: (1, T, P)
    lengths_tensor = torch.tensor([lengths], dtype=torch.long).to(device)           # shape: (1, T)

    with torch.no_grad():
        output = model(input_tensor, lengths_tensor)  # shape: (1, vocab_size)
        probabilities = torch.sigmoid(output)

        # Trả về dạng [(số, xác suất)]
        idx_and_probs = [(idx, prob.item()) for idx, prob in enumerate(probabilities[0])]
        return idx_and_probs
    
def retrain_model(kind: str, place: str):
    # Khởi tạo dataset
    dataset = LotteryDataset(
        data=read_xoso("xs_data/xsmn_data.json", kind, place),
        num_day_steps=num_day_steps,
        max_prizes_per_day=max_prizes_per_day,
        vocab_size=vocab_size
    )

    # Chia dataset với tỷ lệ hợp lý hơn
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    batch_size = 16  # Giảm batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Tổng số mẫu trong dataset: {len(dataset)}")
    print(f"Số mẫu huấn luyện: {len(train_dataset)}")
    print(f"Số mẫu validation: {len(val_dataset)}")
    print(f"Số mẫu kiểm tra: {len(test_dataset)}")

    # Kiểm tra một batch
    for batch_X, _, batch_y in train_loader:
        print(f"\nInput batch X shape: {batch_X.shape}")
        print(f"Output batch y shape: {batch_y.shape}")
        print(f"Số phần tử (không trùng) trong batch y đầu tiên: {batch_y[0].sum().item()}")
        print(f"Batch x đầu tiên: {batch_X[0]}")
        break

    embedding_dim = 32  # Số chiều của vector embedding
    hidden_size = 128   # Số unit ẩn của mỗi layer
    num_layers = 2      # Số layer của mạng
    dropout_rate = 0.3  # Tỉ lệ dropout

    model = LotteryModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_prizes_per_day=max_prizes_per_day,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(device))  # Tăng trọng số cho positive class
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Sử dụng AdamW với weight decay và giảm learning rate xuống 0.001

    num_epochs = 15

    print("Bắt đầu training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, lengths, targets) in enumerate(train_loader):
            inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)
  
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, lengths, targets in val_loader:
                inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)
                outputs = model(inputs, lengths)
                val_loss += criterion(outputs, targets).item()

                predicted_probs = torch.sigmoid(outputs)
                predicted_labels = (predicted_probs > 0.5).float()
                
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        precision = precision_score(all_targets.flatten(), all_predictions.flatten(), zero_division=0)
        recall = recall_score(all_targets.flatten(), all_predictions.flatten(), zero_division=0)
        f1 = f1_score(all_targets.flatten(), all_predictions.flatten(), zero_division=0)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    torch.save(model.state_dict(), f'model/model_{kind}_{place}_{num_epochs}.pth')

    print("Training completed!")

    # Test phase
    model.load_state_dict(torch.load(f'model/model_{kind}_{place}_{num_epochs}.pth'))
    print(f'model/model_{kind}_{place}_{num_epochs}.pth')

    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, lengths, targets in test_loader:
            inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)
            outputs = model(inputs, lengths)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float()

            # Chuyển tensor về numpy, lúc này là list chứa dữ liệu numpy
            test_predictions.extend(predicted_labels.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    # Ép list về thành numpy, giờ là mảng numpy chứa dữ liệu numpy
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    test_precision = precision_score(test_targets.flatten(), test_predictions.flatten(), zero_division=0)
    test_recall = recall_score(test_targets.flatten(), test_predictions.flatten(), zero_division=0)
    test_f1 = f1_score(test_targets.flatten(), test_predictions.flatten(), zero_division=0)

    print(f"\nTest Results:")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    probs_with_indices = predict_next_day(model, dataset.data, device)
    print("--- DỰ ĐOÁN ---")
    dict_num_with_score = dict()
    for idx, prob in probs_with_indices:
        dict_num_with_score[idx] = prob

    dict_num_with_score = dict(sorted(dict_num_with_score.items(), key=lambda item: item[1], reverse=True))

    for number, score in list(dict_num_with_score.items())[:10]:
        print(f"{number}: {score:.2f}")

if __name__ == "__main__":
    retrain_model("lo", "name")