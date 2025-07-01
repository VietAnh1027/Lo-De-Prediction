import requests
from data_reading import read_json

API_KEY = "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"
URL = "https://openrouter.ai/api/v1/chat/completions"
data = read_json("xsmb_data.json")

models = [
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deeepseek-r1-0528-qwen3-8b:free",
    "moonshotai/kiimi-dev-72b:free",
]

content = '''Bạn được cung cấp dữ liệu về các số xuất hiện trong nhiều ngày, dưới dạng ma trận 2 chiều: mỗi hàng là các số trong 1 ngày, mỗi cột là số ở các ngày khác nhau. Dữ liệu: {data}. 
Nhiệm vụ:
- Phân tích dữ liệu này bằng các kỹ thuật xác suất hoặc dựa trên ngữ cảnh các số đã xuất hiện.
- Dự đoán 10 số có xác suất cao nhất sẽ xuất hiện trong ngày tiếp theo.
- Phản hồi duy nhất dưới dạng JSON, trong đó key là con số (chuỗi), value là xác suất xuất hiện (chuỗi, định dạng thập phân, ví dụ: "0.55").
- Nếu key là số có một chữ số, thêm "0" vào đầu (ví dụ: "08")
- Không viết bất kỳ nội dung nào khác ngoài JSON.
'''

messages = [
    {"role":"system", "content": "Bạn là một trợ lý AI thông minh hữu ích"},
    {"role": "user", "content": content}
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response_data = None

for model in models:
    payload = {
        "model": model,
        "messages": messages
    }
    
    try:
        print(f"Trying model: {model}")
        resp = requests.post(URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        response_data = resp.json()
        break  # Nếu thành công thì thoát vòng lặp
    except requests.RequestException as e:
        print(f"Model {model} failed: {e}")
        continue  # Thử model tiếp theo

if response_data:
    print("Response:", response_data["choices"][0]["message"]["content"])
else:
    print("All models failed.")
