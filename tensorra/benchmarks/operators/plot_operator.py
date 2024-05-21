import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

#res = pd.read_csv(sys.argv[1])
res = pd.read_csv("operator_result.csv")

def plot_operator(df, operator_name):
    labels = ["1k", "10k", "100k", "1m", "10m"]
    df = df[df["operator"] == operator_name]
    print(df)
    fig, ax = plt.subplots(figsize=(12, 4))
    #x = np.arange(len(labels))
    x = np.array([1, 2, 3, 4, 5])
    width = 0.2
    df_spark = df[df["framework"] == "spark"]["exec_time"]
    df_pandas = df[df["framework"] == "pandas"]["exec_time"]
    df_monetdb = df[df["framework"] == "monetdb"]["exec_time"]
    df_tensorra = df[df["framework"] == "tensorra"]["exec_time"]
    ax.bar(x - width * 3 / 2, df_monetdb, width, label="monetdb")
    ax.bar(x - width / 2, df_spark, width, label="spark")
    ax.bar(x + width / 2, df_pandas, width, label="pandas")
    ax.bar(x + width * 3 / 2, df_tensorra, width, label="tensorra")
    ax.set_yscale("log")
    ax.set_ylabel('time(ms)')
    ax.set_xlabel('datasize')
    ax.set_title(operator_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(operator_name + ".eps", format='eps', bbox_inches = 'tight') 
    plt.savefig(operator_name + ".jpg", format='jpg', bbox_inches = 'tight') 
    plt.show()

plot_operator(res, "selection")
plot_operator(res, "projection")
plot_operator(res, "inner_join")
plot_operator(res, "groupby")
plot_operator(res, "aggregration")