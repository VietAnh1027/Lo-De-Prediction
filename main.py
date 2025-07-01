from final_prediction import response_predict_nums
from fastapi import FastAPI
from train_lottery_lo import LotteryModel
from data_reading import read_xoso
import torch

app = FastAPI()
data_lo_bac = read_xoso("xs_data/xsmb_data_full.json", "lo", "bac")
data_de_bac = read_xoso("xs_data/xsmb_data_full.json", "de", "bac")

@app.get("/lo")
async def root():
    return response_predict_nums(data_lo_bac, "lo")

@app.get("/de")
async def root():
    return response_predict_nums(data_de_bac, "de")