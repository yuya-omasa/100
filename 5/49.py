import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyDMy_D9WZiZgAIc1h1fisTuIQ3CByLZPbw")

model = genai.GenerativeModel("gemini-1.5-flash")

with open("neko.txt", "r") as f:
    content = f.read()

tokens = model.count_tokens(
    contents = content
)

print(tokens)

with open("out/49.out", "w") as f:
    f.write(str(tokens.total_tokens))