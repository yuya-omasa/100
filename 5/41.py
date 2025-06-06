import google.generativeai as genai
import os
from functools import reduce

genai.configure(api_key="GEMINI-KEY")

model = genai.GenerativeModel("gemini-1.5-flash")

examples = [
    (
        "日本の近代化に関連するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。\n\n"
        "ア　府知事・県令からなる地方官会議が設置された。\n"
        "イ　廃藩置県が実施され，中央から府知事・県令が派遣される体制になった。\n"
        "ウ　すべての藩主が，天皇に領地と領民を返還した。\n\n"
        "解答: ウ→イ→ア\n\n"
    ),
    (
        "江戸幕府の北方での対外的な緊張について述べた次の文ア～ウを年代の古い順に正しく並べよ。\n\n"
        "ア　レザノフが長崎に来航したが，幕府が冷淡な対応をしたため，ロシア船が樺太や択捉島を攻撃した。\n"
        "イ　ゴローウニンが国後島に上陸し，幕府の役人に捕らえられ抑留された。\n"
        "ウ　ラクスマンが根室に来航し，漂流民を届けるとともに通商を求めた。\n\n"
        "解答: ウ→ア→イ\n\n"
    ),
    (
        "中居屋重兵衛の生涯の期間におこったできごとについて述べた次のア～ウを，年代の古い順に正しく並べよ。\n\n"
        "ア　アヘン戦争がおこり，清がイギリスに敗北した。\n"
        "イ　異国船打払令が出され，外国船を撃退することが命じられた。\n"
        "ウ　桜田門外の変がおこり，大老の井伊直弼が暗殺された。\n\n"
        "解答: イ→ア→ウ\n\n"
    ),
    (
        "加藤高明が外務大臣として提言を行ってから、内閣総理大臣となり演説を行うまでの時期のできごとについて述べた次のア～ウを，年代の古い順に正しく並べよ。\n\n"
        "ア　朝鮮半島において，独立を求める大衆運動である三・一独立運動が展開された。\n"
        "イ　関東大震災後の混乱のなかで，朝鮮人や中国人に対する殺傷事件がおきた。\n"
        "ウ　日本政府が，袁世凱政府に対して二十一カ条の要求を突き付けた。\n\n"
        "解答: ウ→ア→イ\n\n"
    ),
]

question = (
    "9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。\n\n"
    "ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。\n"
    "イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。\n"
    "ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。\n"
)

contents = reduce(lambda x, y: x + y, examples) + question

response = model.generate_content(
    contents = contents
)

print(response.text)

with open("out/41.out","w") as f:
    f.write(response.text)