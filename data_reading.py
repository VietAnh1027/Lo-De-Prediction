import json

def read_xoso(namefile: str, kind: str, place: str):
    with open(namefile, "r") as f:
        data = json.load(f)

    all_numbers = []

    if kind == "lo":
        if place == "bac":
            for day in data:
                all_results = day.get("all_results", {})
                day_prize = []
                for prize in all_results:
                    if prize == "ĐB":
                        continue
                    day_prize.extend([int(x[-2:]) for x in all_results[prize]])
                day_prize.sort()
                all_numbers.append(day_prize)

        elif place == "nam":
            for day in data:
                all_results = day.get("all_results", {})
                day_prize = []
                for city in all_results:
                    for prize in all_results[city]:
                        if prize == "ĐB":
                            continue
                        day_prize.extend([int(x[-2:]) for x in all_results[city][prize] if x.isdigit()])
                day_prize.sort()
                all_numbers.append(day_prize)
        else:
            print("Miền không hợp lệ!")

    elif kind == "de":
        if place == "bac":
            for day in data:
                all_numbers.append(int((day.get("special_prize")).strip()[-2:]))

        elif place == "nam":
            for day in data:
                special_prizes = [int((prize).strip()[-2:]) for prize in day.get("special_prize") if prize.isdigit()]
                special_prizes.sort()
                all_numbers.append(special_prizes)
        else:
            print("Miền không hợp lệ!")
    
    else:
            print("Giải không hợp lệ!")

    return all_numbers


if __name__ == "__main__":
    # print(read_xoso("xsmb_data_full.json", "lo", "bac")[-10:])
    print(read_xoso("xs_data/xsmn_data.json", "de", 'nam')[-1:])
