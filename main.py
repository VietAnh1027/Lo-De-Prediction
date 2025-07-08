# from final_prediction import response_predict_nums
from fastapi import FastAPI
from pydantic import BaseModel, RootModel
from typing import List, Dict
from train_lottery_lo import train_predict_lo_bac
from train_lottery_de import train_predict_de_bac
from train_lottery_others import train_predict_others
from data_reading import read_xoso
import os
import json
from json.decoder import JSONDecodeError

class BacNewPrize(BaseModel):
    date: str
    special_prize: str
    all_results: Dict[str, List[str]]

class ProvinceResults(RootModel[Dict[str, List[str]]]):
    pass

class OtherNewPrize(BaseModel):
    date: str
    special_prize: List[str]
    all_results: Dict[str, ProvinceResults] 

app = FastAPI()

data_bac_path = "xs_data/xsmb_data.json"
data_nam_path = "xs_data/xsmn_data.json"
data_trung_path = "xs_data/xsmt_data.json"

# Trả về 10 số xác suất vào ngày tiếp theo dưới dạng JSON
@app.get("/lo-mien-bac")
async def root():
    return train_predict_lo_bac()

@app.get("/de-mien-bac")
async def root():
    return train_predict_de_bac()

@app.get("/lo-mien-nam")
async def root():
    return train_predict_others(data_path=data_nam_path , kind="lo", place="nam", epoch=15)

@app.get("/lo-mien-trung")
async def root():
    return train_predict_others(data_path=data_trung_path, kind="lo", place="trung", epoch=15)

@app.get("/de-mien-nam")
async def root():
    return train_predict_others(data_path=data_nam_path, kind="de", place="nam", epoch=45)

@app.get("/de-mien-trung")
async def root():
    return train_predict_others(data_path=data_trung_path, kind="de", place="trung", epoch=45)

# Thêm dữ liệu ngày mới vào dữ liệu bắc, trung, nam
@app.post("/add-bac")
def add_data_bac(item: BacNewPrize):
    if os.path.exists(data_bac_path) and os.path.getsize(data_bac_path) > 0:
        try:
            with open(data_bac_path, "r") as f:
                current_data = json.load(f)
        except JSONDecodeError:
            current_data = []
    else:
        current_data = []

    current_data.append(item.model_dump())

    with open(data_bac_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    return {"message": "Thêm thành công", "data": item.model_dump()}

@app.post("/add-nam")
def add_data_nam(item: OtherNewPrize):
    if os.path.exists(data_nam_path) and os.path.getsize(data_nam_path) > 0:
        try:
            with open(data_nam_path, "r") as f:
                current_data = json.load(f)
        except JSONDecodeError:
            current_data = []
    else:
        current_data = []

    current_data.append(item.model_dump())

    with open(data_nam_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    return {"message": "Thêm thành công", "data": item.model_dump()}

@app.post("/add-trung")
def add_data_nam(item: OtherNewPrize):
    if os.path.exists(data_trung_path) and os.path.getsize(data_trung_path) > 0:
        try:
            with open(data_trung_path, "r") as f:
                current_data = json.load(f)
        except JSONDecodeError:
            current_data = []
    else:
        current_data = []

    current_data.append(item.model_dump())

    with open(data_trung_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    return {"message": "Thêm thành công", "data": item.model_dump()}