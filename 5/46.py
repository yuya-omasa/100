import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyDMy_D9WZiZgAIc1h1fisTuIQ3CByLZPbw")

model = genai.GenerativeModel("gemini-1.5-flash")

contents = (
    "季節を感じさせる日本の川柳を10個作ってください。"
    "風景や人間の心情を表現してください。"
)

response = model.generate_content(
    contents = contents
)

print(response.text)

with open("out/46.out", "w") as f:
    f.write(response.text)