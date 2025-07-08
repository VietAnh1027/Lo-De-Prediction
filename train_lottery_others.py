import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from data_reading import read_xoso

# TRAIN MODEL, DỰ ĐOÁN

torch.manual_seed(42)
num_day_steps = 10  # Dự đoán dựa trên 10 ngày gần nhất
# max_prizes_per_day  # Số lượng số trúng thưởng tối đa trong một ngày
vocab_size = 101    # Số lượng số khả thi: 0 - 99 + padding token (100)
padding_value = 100  # Giá trị padding

class LotteryDataset(Dataset):
    def __init__(self, data, num_day_steps, max_prizes_per_day, vocab_size, padding_value):
        self.data = data
        self.num_day_steps = num_day_steps
        self.max_prizes_per_day = max_prizes_per_day
        self.vocab_size = vocab_size
        self.padding_value = padding_value

        self.X = []
        self.y = []

        self._prepare_data()

    def _prepare_data(self):
        # Padding dữ liệu để mỗi ngày có đủ max_prizes_per_day số
        padded_data = []
        for day_data in self.data:
            padded_day = list(day_data)
            # Padding nếu không đủ số
            while len(padded_day) < self.max_prizes_per_day:
                padded_day.append(self.padding_value)
            # Cắt bớt nếu quá nhiều số (tránh trường hợp vượt quá max)
            padded_day = padded_day[:self.max_prizes_per_day]
            padded_data.append(padded_day)
        
        # Tạo sequences
        for i in range(len(padded_data) - self.num_day_steps):
            input_sequence = padded_data[i:i + self.num_day_steps]
            self.X.append(torch.tensor(input_sequence, dtype=torch.long))

            # Tạo target multi-hot vector (chỉ tính các số từ 0-99, bỏ qua padding)
            target_next_day = padded_data[i + self.num_day_steps]
            target_multi_hot = torch.zeros(100, dtype=torch.float32)  # Chỉ 100 số (0-99)
            for num in target_next_day:
                if num < 100:  # Bỏ qua padding value
                    target_multi_hot[num] = 1.0
            self.y.append(target_multi_hot)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
class LotteryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_prizes_per_day, hidden_size, num_layers=1, dropout_rate=0.0):
        super(LotteryModel, self).__init__()

        self.max_prizes_per_day = max_prizes_per_day
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer với padding_idx để ignore padding tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=100)

        self.embedding_bn = nn.BatchNorm1d(max_prizes_per_day * embedding_dim)

        self.lstm = nn.LSTM(
            input_size=max_prizes_per_day * embedding_dim,
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
        self.fc3 = nn.Linear(hidden_size // 2, 100)  # Output 100 số (0-99)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        batch_size, num_day_steps, num_prize = x.size()
        x_flatten = x.view(-1)

        embedded_x = self.embedding(x_flatten)
        lstm_input = embedded_x.view(batch_size, num_day_steps, num_prize * self.embedding_dim)
        lstm_input_norm = self.embedding_bn(lstm_input.transpose(1, 2)).transpose(1, 2)
        lstm_output, (h_n, c_n) = self.lstm(lstm_input_norm)

        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)

        combined_output = lstm_output + attn_output

        final_output = combined_output[:, -1, :]

        x1 = self.dropout(self.gelu(self.fc1(final_output)))
        x1 = self.layer_norm1(x1 + final_output)  # Residual connection
        
        x2 = self.dropout(self.gelu(self.fc2(x1)))
        x2 = self.layer_norm2(x2)
        
        output = self.fc3(x2)
        
        return output
    
def predict_next_day(model, last_days_data, device, max_prizes_per_day):
    model.eval()
    
    # Lấy num_day_steps ngày gần nhất
    recent_data = last_days_data[-num_day_steps:]
    
    # Padding dữ liệu
    padded_input = []
    for day_data in recent_data:
        padded_day = list(day_data)
        # Padding nếu không đủ số
        while len(padded_day) < max_prizes_per_day:
            padded_day.append(padding_value)
        # Cắt bớt nếu quá nhiều số
        padded_day = padded_day[:max_prizes_per_day]
        padded_input.append(padded_day)
    
    # Thêm batch dimension
    input_tensor = torch.tensor([padded_input], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output)
        # probabilities[0]: Lấy ra batch size đầu tiên và duy nhất
        idx_and_probs = dict()
        for idx, prob in enumerate(probabilities[0]):
            idx_and_probs[idx] = prob.item()

        idx_and_probs = dict(sorted(idx_and_probs.items(), key=lambda item: item[1], reverse=True)[:10])
        
        return idx_and_probs
    
def train_predict_others(data_path, kind, place, epoch=15):
    # Đọc dữ liệu
    data = read_xoso(data_path, kind, place)
    max_prizes_per_day = 0
    if place == "nam":
        if kind == "de":
            max_prizes_per_day= 4
        elif kind == "lo":
            max_prizes_per_day= 68
        else:
            print("Giải không hợp lệ!")
    
    elif place == "trung":
        if kind == "de":
            max_prizes_per_day= 3
        elif kind == "lo":
            max_prizes_per_day= 51
        else:
            print("Giải không hợp lệ!")
    else:
        print("Miền không hợp lệ!")
    
    # Khởi tạo dataset
    dataset = LotteryDataset(
        data=data,
        num_day_steps=num_day_steps,
        max_prizes_per_day=max_prizes_per_day,
        vocab_size=vocab_size,
        padding_value=padding_value
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
    for batch_X, batch_y in train_loader:
        print(f"\nInput batch X shape: {batch_X.shape}")
        print(f"Output batch y shape: {batch_y.shape}")
        print(f"Batch X đầu tiên: {batch_X[0]}")
        print(f"Batch y đầu tiên: {batch_y[0]}")
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

    num_epochs =epoch

    print(f"--- Dữ liệu là kết quả xổ số miền {place} giải {kind} ---")
    print("Bắt đầu training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
  
            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
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

    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
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

    print(f"--- DỰ ĐOÁN {kind} MIỀN {place} THÀNH CÔNG! ---")
    nums_and_probs = predict_next_day(model, data, device, max_prizes_per_day)
    return nums_and_probs

if __name__ == "__main__":
    epoch = 45
    kind = "de"
    place = "nam"
    nums_and_probs = train_predict_others("xs_data/xsmt_data.json",kind=kind,place=place,epoch=epoch)
    print(f"--- KẾT QUẢ DỰ ĐOÁN VỚI {epoch} EPOCHS ---")
    for num, prob in nums_and_probs.items():
        print(f"{num}: {prob:.2f}")