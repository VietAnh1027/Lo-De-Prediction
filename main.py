# from final_prediction import response_predict_nums
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from train_lottery_lo import train_predict_lo_bac
from train_lottery_de import train_predict_de_bac
from train_lottery_others import train_predict_others
from data_reading import read_xoso
import torch
import json

class BacNewPrize(BaseModel):
    date: str
    special_prize: str
    all_results: Dict[str, List[str]]

class OtherNewPrize(BaseModel):
    date: str
    special_prize: List[str]
    all_results: Dict[
        Dict[str, List[str]]
    ]

app = FastAPI()
data_lo_bac = read_xoso("xs_data/xsmb_data.json", "lo", "bac")
data_de_bac = read_xoso("xs_data/xsmb_data.json", "de", "bac")

@app.get("/lo-mien-bac")
async def root():
    return train_predict_lo_bac(data_lo_bac, "lo")

@app.get("/de-mien-bac")
async def root():
    return train_predict_de_bac(data_de_bac, "de")

@app.get("/lo-mien-nam")
async def root():
    return train_predict_others(data_path="xsmn_test.json", kind="lo", place="nam", epoch=15)

@app.get("/lo-mien-trung")
async def root():
    return train_predict_others(data_path="xsmt_test.json", kind="lo", place="trung", epoch=15)

@app.get("/de-mien-nam")
async def root():
    return train_predict_others(data_path="xsmn_test.json", kind="de", place="nam", epoch=45)

@app.get("/de-mien-trung")
async def root():
    return train_predict_others(data_path="xsmt_test.json", kind="de", place="trung", epoch=45)

@app.post("/add-bac")
def add_data_bac(item: BacNewPrize):
    with open("xsmb_test.json", "r") as f:
        current_data = json.load(f)

    current_data.append(item.model_dump())

    with open("xsmb_test.json", "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    return {"message": "Thêm thành công", "data": item.model_dump()}

@app.post("/add-nam")
def add_data_nam(item: OtherNewPrize):
    with open("xsmn_test.json", "r") as f:
        current_data = json.load(f)

    current_data.append(item.model_dump())

    with open("xsmb_test.json", "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    return {"message": "Thêm thành công", "data": item.model_dump()}