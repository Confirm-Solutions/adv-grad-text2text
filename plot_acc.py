import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("results.csv")

plt.plot(df["attention_block"], df["token_acc"])
plt.xticks(df["attention_block"])
plt.title("Token ACC Plot on GPT-2")
plt.xlabel("Attention Block")
plt.ylabel("Token ACC")
plt.savefig("figure.png", dpi=400)