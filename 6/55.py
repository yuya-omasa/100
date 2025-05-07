import pandas as pd

# read_csv に sep="," とエラー処理オプションを追加
df = pd.read_csv("out/54.out", header=None, sep=",", on_bad_lines="skip")

# semantic / syntactic 分類
sem = df[~df[0].str.contains("gram", na=False)]
syn = df[df[0].str.contains("gram", na=False)]

# 正解率の計算（第5列と第6列を比較）
print("Acc of semantic: ", (sem[4] == sem[5]).sum() / len(sem))
print("Acc of syntactic: ", (syn[4] == syn[5]).sum() / len(syn))
