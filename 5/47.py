import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyDMy_D9WZiZgAIc1h1fisTuIQ3CByLZPbw")

model = genai.GenerativeModel("gemini-1.5-flash")

contents = (
    "あなたは日本の歌人です。次の川柳の面白さを10段階で評価してください。出力は必ず1から10の数字のみで答えること。"
)

haiku = [
    "初雪かな　屋根白くして　眠りけり",
    "春の芽吹き　土の匂い深し　希望かな",
    "蛍光る　夏の夜空に　消え入りぬ",
    "蝉しぐれ　煩きも心地　夏の昼",
    "秋風や　金木犀の香　胸に沁む",
    "紅葉燃ゆ　山肌赤々と　燃え尽きて",
    "落ち葉舞い　冬の足音　近づくかな",
    "雪解け水　清き流れに　心洗う",
    "梅雨空や　洗濯物干せず　ため息つく",
    "夏の雲　悠々と流れ　どこまでも",
]

for i in range(len(haiku)):
    response = model.generate_content(
        contents = contents + haiku[i]
    )

    print(response.text.replace("\n", ""))

    with open("out/47.out", "a") as f:
        f.write(response.text)

