import json
import csv

# Sub_task 1
with open("../output/submission1.jsonl", "r", encoding="utf-8") as fin, \
     open("submission_1.csv", "w", newline="", encoding="utf-8") as fout:

# Sub_task 2
# with open("../output/submission2.jsonl", "r", encoding="utf-8") as fin, \
#      open("submission_2.csv", "w", newline="", encoding="utf-8") as fout:

    writer = csv.DictWriter(fout, fieldnames=["id", "translation"])
    writer.writeheader()

    for line in fin:
        data = json.loads(line)
        writer.writerow({
            "id": data["id"],
            "translation": data["pred"]
        })
