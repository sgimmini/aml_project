import seaborn as sns 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
sns.set_style(style="darkgrid")

df_full = pd.DataFrame()
for c in os.listdir("TB_csvs/triple/"):
    df = pd.read_csv(os.path.join("TB_csvs/triple", c))
    df.drop("Wall time", axis=1, inplace=True)
    df["Model"] = str(c.split(".")[0])
    df = df.rename(columns={"Value": "Loss", "Step": "Epoch"})

    df_full = pd.concat([df_full, df])

print(df_full)

g = sns.lineplot(x="Epoch", y="Loss",
             hue="Model",
             data=df_full)

fig = g.get_figure()
plt.tight_layout()
fig.savefig("test.png")
