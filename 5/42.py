import google.generativeai as genai
import os
import requests
import csv
from tqdm import tqdm
import time
from typing import Any

url = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/refs/heads/main/JMMLU/high_school_computer_science.csv"
filename = "data/high_school_computer_science.csv"
urlData = requests.get(url).content

with open(filename, mode="wb") as f:
    f.write(urlData)

with open(filename, "r", encoding = "UTF-8") as f:
    data = csv.reader(f)
    data = [list(item) for item in data]

genai.configure(api_key = "GEMINI-KEY")

model = genai.GenerativeModel("gemini-1.5-flash")

def preprocess(problemlist: list[str]) -> dict[str, str]:
    problem = problemlist[0]
    choices = problemlist[1:5]
    answer = problemlist[5]
    content = (
        f"{problem}\n\n"
        f"A: {choices[0]}\n"
        f"B: {choices[1]}\n"
        f"C: {choices[2]}\n"
        f"D: {choices[3]}\n"
    )
    return {"content": content, "answer": answer}

problems = map(preprocess, data)
responses: list[Any] =list()
accuracy = list()

count = 0

for problem in tqdm(problems):
    contents = (
        "以下の高校情報科学に関する問いに答えなさい。出力は必ずA,B,C,Dの記号のみ簡潔に出力しなさい。\n\n"
        f"{problem['content']}"
    )

    response = model.generate_content(
        contents = contents
    )

    response = response.text.replace("\n", "")

    responses.append(response)

    true_or_false = 1 if response == problem["answer"] else 0
    accuracy.append(true_or_false)

    count += 1
    if (count % 15 == 0):
        count = 0
        time.sleep(61)

with open("out/42.model.out", "w") as f:
    for text in responses:
        f.write(text + "\n")

acc = sum(accuracy) / len(accuracy)
print(f"Accuracy: {acc}\n")

with open("out/42.out", "w") as f:
    f.write(f"Accuracy: {acc}\n")