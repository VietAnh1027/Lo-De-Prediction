import numpy as np
from data_reading import read_xoso
from train_lottery_lo import LotteryModel, predict_next_day, retrain_model
from train_lottery_de import SpecialPrizeModel, predict_special_prize, retrain_special
import torch
import os

# KẾT HỢP THÊM PHƯƠNG PHÁP XÁC SUẤT

# NUM_RETRAIN = 0

def freq(data, n):
    data = np.array(data).flatten()
    count_n = np.count_nonzero(data == n)
    return float(count_n / len(data))

def bayes(history, n):
    epsilon = 1e-10
    count_total = 0
    count_occurrences = 0
    for i in range(len(history) - 1):
        next_day = history[i+1]
        count_total += 1
        if n in next_day:
            count_occurrences += 1
    probability = (count_occurrences + epsilon) / (count_total + epsilon)
    return probability

def response_predict_nums(data, type):
    # global NUM_RETRAIN
    # if not os.path.exists("count.txt"):
    #     with open("count.txt", "w") as f:
    #         f.write("0")
    # with open("count.txt", "r") as f:
    #     NUM_RETRAIN = int(f.read())
    # with open("count.txt", "w") as f:
    #     if NUM_RETRAIN == 20:
    #         retrain_model()
    #         retrain_special()
    #         NUM_RETRAIN = 0
    #     else:
    #         NUM_RETRAIN += 1
    #     f.write(str(NUM_RETRAIN))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dict_num_with_score = dict()
    alpha = 0
    beta = 0
    gamma = 1 - alpha - beta

    if type == "lo":
        model = LotteryModel(
            vocab_size=100,
            embedding_dim=32,
            num_prizes_per_day=26,
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.3
        )
        model.load_state_dict(torch.load(f'model/lottery_lstm_model_15.pth'))
        model.to(device)
        numbers_with_probs = predict_next_day(model, data, device)
    
    if type == "de":
        model = SpecialPrizeModel(
            vocab_size=100,
            embedding_dim=32,
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.3
        )
        model.load_state_dict(torch.load(f'model/special_lottery_model_45.pth'))
        model.to(device)
        numbers_with_probs = predict_special_prize(model, data, device)

    for number, score in numbers_with_probs:
        # score = alpha*freq(data, number) + beta*bayes(data, number) + gamma*prob
        number = str(number)
        if len(number) == 1: 
            number = "0" + number
        dict_num_with_score[number] = round(score,2)
    dict_num_with_score = dict(sorted(dict_num_with_score.items(), key=lambda item: item[1], reverse=True))
    return dict(list(dict_num_with_score.items())[:10])

if __name__ == "__main__":
    data = read_xoso("xs_data/xsmb_data.json", "lo", "bac")

    # Khởi tạo model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LotteryModel(
            vocab_size=100,
            embedding_dim=32,
            num_prizes_per_day=26,
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.3
        )
    model.load_state_dict(torch.load(f'model/lottery_lstm_model_15.pth'))
    model.to(device)

    dict_num_with_score = response_predict_nums(model, device, data)

    for number, score in list(dict_num_with_score.items())[:10]:
        print(f"{number}: {round(score,2)}")
