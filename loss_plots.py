import seaborn as sns 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
sns.set_style(style="darkgrid")

choice = "binary"

df_full = pd.DataFrame()
for c in os.listdir(f"TB_csvs/{choice}/"):
    df = pd.read_csv(os.path.join(f"TB_csvs/{choice}", c))
    df.drop("Wall time", axis=1, inplace=True)
    df["Model"] = str(c.split(".")[0])
    df = df.rename(columns={"Value": "Loss", "Step": "Epoch"})

    #df = df.drop(df[df.Epoch > 20].index)

    df_full = pd.concat([df_full, df])

print(df_full)

g = sns.lineplot(x="Epoch", y="Loss",
             hue="Model",
             data=df_full)

fig = g.get_figure()
plt.tight_layout()
fig.savefig("test.png")
