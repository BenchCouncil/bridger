#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from matplotlib.patches import Ellipse
#from scipy.stats import gmean, gstd
#from matplotlib.pyplot import thetagrids
pd.set_option('display.max_rows', None)

def plot_operator(df, operator_name):
    labels = ["1k", "10k", "100k", "1m", "10m"]
    df = df[df["operator"] == operator_name]
    fig, ax = plt.subplots(figsize=(12, 4))
    #x = np.arange(len(labels))
    x = np.array([1, 2, 3, 4, 5])
    width = 0.2
    df_spark = df[df["framework"] == "spark"]["exec_time"]
    df_pandas = df[df["framework"] == "pandas"]["exec_time"]
    df_monetdb = df[df["framework"] == "monetdb"]["exec_time"]
    df_tensorra = df[df["framework"] == "tensorra"]["exec_time"]
    ax.bar(x - width * 3 / 2, df_monetdb, width, label="MonetDB")
    ax.bar(x - width / 2, df_spark, width, label="Spark", hatch="..")
    ax.bar(x + width / 2, df_pandas, width, label="pandas", hatch="//")
    ax.bar(x + width * 3 / 2, df_tensorra, width, label="TensorTable", hatch="\\\\")
    ax.set_yscale("log")
    ax.set_ylabel('time(ms)')
    ax.set_xlabel('datasize')
    ax.set_title(operator_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(operator_name + ".eps", format='eps', bbox_inches = 'tight') 

def clean(x):
    iqr = x.quantile(0.75) - x.quantile(0.25)
    x_lower_bound = x.quantile(0.25) - 1.5 * iqr
    x_upper_bound = x.quantile(0.75) + 1.5 * iqr
    res_dic = {} 
    for col_index in x.columns:
        if(col_index not in {"framework","pipeline","datasize","operator","row_num"}):
            sum = 0 
            count = 0 
            for row_index in x.index:
                tmp = x[col_index].loc[row_index]
                if(tmp <= x_upper_bound[col_index] and tmp >= x_lower_bound[col_index]):
                    sum = sum + tmp
                    count = count + 1
            if(count == 0):
                avg = 0
            else:
                avg = sum / count 
            res_dic[col_index] = avg
    return pd.Series(res_dic)

def plot_relative_performance(figname, df):
    if(figname == "lr_relative"):
        labels = ["1k", "10k", "100k", "1m", "10m"]
    elif(figname == "cov_relative"):
        labels = ["100", "1k", "10k", "100k", "1m"]
    else:
        print("Unkown figure") 
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.array([1, 2, 3, 4, 5])
    width = 0.25
    df_spark = df[df["framework"] == "spark"]["total"].reset_index()["total"] #/ df[df["framework"] == "tensorra"]["total"]
    df_pandas = df[df["framework"] == "pandas"]["total"].reset_index()["total"]  #/ df[df["framework"] == "tensorra"]["total"]
    df_tensorra = df[df["framework"] == "tensorra"]["total"].reset_index()["total"]  #/ df[df["framework"] == "tensorra"]["total"]
    df_spark = df_spark / df_tensorra
    df_pandas = df_pandas / df_tensorra
    df_tensorra = df_tensorra / df_tensorra
    print(df_spark)
    print(df_pandas)
    df_spark = df_spark.apply(lambda x : 1/x)
    df_pandas = df_pandas.apply(lambda x : 1/x)
    ax.bar(x - width, df_pandas, width, label="pandas", hatch="..")
    ax.bar(x, df_spark, width, label="Spark")
    ax.bar(x + width, df_tensorra, width, label="TensorTable", hatch="//")
    ax.axhline(y=1, ls="--", c="r")
    ax.set_ylabel('relative performance')
    ax.set_xlabel('datasize')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(figname + ".eps", format='eps', bbox_inches = 'tight') 
    return 0

def plot_stack_bar(figname, df, first):
    if(first == True):
        plt.figure(figsize=(4,4.22))
    else:
        plt.figure(figsize=(4,4))
    ind = [1, 2, 3]
    plt.xticks(ind, ("pandas", "Spark", "TensorTable"))
    plt.yscale("log")
    conversion = df["conversion_time"].values
    linear = df["linear_time"].values
    relational = df["relational_time"].values
    width = 0.25
    plt.bar([0.75,1.75,2.75], conversion, width=width, color="c", label="conversion")
    plt.bar([1, 2, 3], linear, width=width, hatch="//", label="linear")
    plt.bar([1.25, 2.25, 3.25], relational, width=width, color="g", hatch="..", label="relational")
    if(first == True):
        plt.ylabel("execution time (s)")
        plt.legend(loc="upper left")
    plt.savefig(figname + ".eps", format='eps', bbox_inches = 'tight') 
    return 0

def plot_linear_regression(df):
    df_1k = df[df["datasize"] == 1000][["framework", "conversion_time", "linear_time", "relational_time"]]
    df_100k = df[df["datasize"] == 100000][["framework", "conversion_time", "linear_time", "relational_time"]]
    df_10m = df[df["datasize"] == 10000000][["framework", "conversion_time", "linear_time", "relational_time"]]
    plot_stack_bar("lr_1k",df_1k,True)
    plot_stack_bar("lr_100k",df_100k,False)
    plot_stack_bar("lr_10m", df_10m,False)
    return 0

def plot_conference_covariance(df):
    df_100 = df[df["datasize"] == 100][["framework", "conversion_time", "linear_time", "relational_time"]]
    df_10k = df[df["datasize"] == 10000][["framework", "conversion_time", "linear_time", "relational_time"]]
    df_1m = df[df["datasize"] ==  1000000][["framework", "conversion_time", "linear_time", "relational_time"]]
    plot_stack_bar("cov_100",df_100,True)
    plot_stack_bar("cov_10k",df_10k,False)
    plot_stack_bar("cov_1m",df_1m,False)
    return 0

def plot_framework(filename):
    df = pd.read_csv(filename)
    df_avg = df.groupby(["framework", "pipeline", "datasize"]).apply(clean).reset_index()
    df_avg["total"] = df_avg["conversion_time"] + df_avg["linear_time"] + df_avg["relational_time"]
    #print(df_avg)
    df_lr = df_avg[df_avg["pipeline"] == "linear_regression"]
    df_cov = df_avg[df_avg["pipeline"] == "covariance"]
    plot_linear_regression(df_lr)
    plot_conference_covariance(df_cov)
    plot_relative_performance("cov_relative", df_cov)
    plot_relative_performance("lr_relative", df_lr)
    return 0

def cal_relative(df, name):
    tensorra = df[df["framework"] == "tensorra"].reset_index()
    res = df[df["framework"] == name].reset_index()
    res[name] = res["exec_time"] / tensorra["exec_time"]
    res = res[["operator", "row_num", name]].groupby(["operator"])[name].apply(gmean).reset_index()
    return res[name]

def plot_motivation_bar(res_pipeline, res_operator):
    """
    Plot motivation
    """
    res_operator.loc[len(res_operator)] = ["monetdb", "addition", 0, 0]
    res_operator.loc[len(res_operator)] = ["monetdb", "matmul", 0, 0]
    res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index() 
    column_map = {"pipeline":"operator", "datasize":"row_num", "linear_time":"exec_time"}
    res_pipeline = res_pipeline.groupby(["framework", "pipeline", "datasize"]).apply(clean).reset_index()[["framework","pipeline","datasize","linear_time"]].rename(columns=column_map)
    res_pipeline["exec_time"] = res_pipeline["exec_time"] * 1000
    pytorch_operator = res_operator[res_operator["framework"] == "pandas"]
    pytorch_operator["framework"] = "pytorch"
    pytorch_operator["exec_time"] = 0
    monetdb_pipeline = res_pipeline[res_pipeline["framework"] == "pandas"]
    monetdb_pipeline["framework"] = "monetdb"
    monetdb_pipeline["exec_time"] = 0
    pytorch_pipeline = res_pipeline[res_pipeline["framework"] == "tensorra"]
    pytorch_pipeline["framework"] = "pytorch"
    res_operator = pd.concat([res_operator, res_pipeline, monetdb_pipeline, pytorch_operator, pytorch_pipeline], axis=0)
    #res_operator = res_operator[res_operator["row_num"].isin([1000, 10000, 100000, 1000000])]
    res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index() 
    res_operator = res_operator.groupby(["framework", "operator"])["exec_time"].apply(gmean).reset_index()
    res_pandas = res_operator[res_operator["framework"]=="pandas"]["exec_time"].reset_index().rename(columns={"exec_time":"pandas"})
    res_spark = res_operator[res_operator["framework"]=="spark"]["exec_time"].reset_index().rename(columns={"exec_time":"spark"})
    res_monetdb = res_operator[res_operator["framework"]=="monetdb"]["exec_time"].reset_index().rename(columns={"exec_time":"monetdb"})
    res_pytorch = res_operator[res_operator["framework"]=="pytorch"]["exec_time"].reset_index().rename(columns={"exec_time":"pytorch"})
    res_tensorra = res_operator[res_operator["framework"]=="tensorra"]["exec_time"].reset_index().rename(columns={"exec_time":"tensorra"})
    operator = res_operator.sort_values(by=["operator"])["operator"].drop_duplicates().reset_index().drop(["index"], axis=1)
    res_relative = pd.concat([operator, res_pandas, res_spark, res_monetdb, res_pytorch, res_tensorra], axis=1).drop(["index"], axis=1)
    print(res_relative)
    res_relative.to_csv("relational.csv",index=False)
    order = ["selection", "projection", "inner_join", "groupby", "addition", "matmul", "covariance", "linear_regression"] 
    res_relative.index = res_relative["operator"]
    res_relative = res_relative.loc[order]

    #Add blank to adjust position
    operators = [
        "Selection", 
        "Projection", 
        "Inner Join", 
        "Group By", 
        "Addition", 
        "MatMul", 
        "MatCov", 
        "LinearReg"
        ]
    plt.figure(figsize=(20,8))
    ind = np.arange(len(operators))
    width = 1 / 6
    def add_cross(position, plt):
        for i in position:
            plt.scatter(i, 2, marker="x", color="r", s=64)
    plt.bar(ind-2*width, res_relative["monetdb"].values, width=width, color="c", label="monetdb")
    plt.bar(ind-width, res_relative["pandas"].values, width=width, label="pandas", hatch="..")
    plt.bar(ind, res_relative["spark"].values, width=width, label="spark", hatch="//")
    pytorch_value = res_relative["tensorra"].values.copy()
    pytorch_value[0:4] = 0
    print(pytorch_value)
    plt.bar(ind+width, pytorch_value, width=width, label="pytorch", hatch="\\\\")
    plt.bar(ind+2*width, res_relative["tensorra"].values, width=width, label="tensorra", hatch="xx")
    plt.xlim(-3*width, 7+3*width)
    plt.xticks(ind, operators, fontsize=18)
    plt.ylabel("execution time (ms)", fontsize=20)
    plt.yscale("log")
    plt.legend(labels=("MonetDB", "pandas", "Spark", "PyTorch", "TensorTable"), loc="upper left", fontsize=18)
    add_cross([width, 1+width, 2+width, 3+width, 4-2*width, 5-2*width, 6-2*width, 7-2*width] ,plt)
    plt.savefig("motivation_bar.eps", format='eps', bbox_inches = 'tight') 
    #plt.show()
    return 0
    
def plot_motivation_radar(res_pipeline, res_operator):
    """
    Plot motivation
    """
    res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index() 
    column_map = {"pipeline":"operator", "datasize":"row_num", "linear_time":"exec_time"}
    res_pipeline = res_pipeline.groupby(["framework", "pipeline", "datasize"]).apply(clean).reset_index()[["framework","pipeline","datasize","linear_time"]].rename(columns=column_map)
    pytorch_operator = res_operator[res_operator["framework"] == "pandas"]
    pytorch_operator["framework"] = "pytorch"
    pytorch_operator["exec_time"] = 0
    monetdb_pipeline = res_pipeline[res_pipeline["framework"] == "pandas"]
    monetdb_pipeline["framework"] = "monetdb"
    monetdb_pipeline["exec_time"] = 0
    pytorch_pipeline = res_pipeline[res_pipeline["framework"] == "tensorra"]
    pytorch_pipeline["framework"] = "pytorch"
    res_operator = pd.concat([res_operator, res_pipeline, monetdb_pipeline, pytorch_operator, pytorch_pipeline], axis=0)
    res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index() 
    print(res_operator)
    #res_operator = res_operator[res_operator["row_num"].isin([100000, 1000000, 10000000])]
    res_pandas = cal_relative(res_operator, "pandas")
    res_spark = cal_relative(res_operator, "spark")
    res_monetdb = cal_relative(res_operator, "monetdb")
    res_tensorra = cal_relative(res_operator, "tensorra")
    res_pytorch = cal_relative(res_operator, "pytorch") 
    operator = res_operator["operator"].drop_duplicates().reset_index()
    res_relative = pd.concat([operator, res_pandas, res_spark, res_monetdb, res_pytorch, res_tensorra], axis=1).drop(["index"], axis=1)
    res_relative[["pandas", "spark", "monetdb", "pytorch", "tensorra"]] = 1 / res_relative[["pandas", "spark", "monetdb", "pytorch", "tensorra"]]
    res_relative.replace([np.inf, -np.inf], 0, inplace=True)
    res_max = res_relative.max(axis=1)
    for f in ["pandas", "spark", "monetdb", "pytorch", "tensorra"]:
        res_relative[f] = res_relative[f] / res_max
    res_relative.loc[len(res_relative)] = ["addition", 0.6, 0.3, 0.2, 1, 1]
    res_relative.loc[len(res_relative)] = ["matmul", 0.5, 0.2, 0.1, 1, 1]
    order = ["selection", "projection", "inner_join", "groupby", "addition", "matmul", "covariance", "linear_regression"] 
    res_relative.index = res_relative["operator"]
    res_relative = res_relative.loc[order]
    res_relative["pytorch"] = res_relative["pytorch"] * 0.98
    print(res_relative)

    #Add blank to adjust position
    operators = [
        " "*8 + "selection", 
        " "*8 + "projection", 
        "inner join", 
        "group by" + " "*5, 
        "addition" + " "*7, 
        "\nmatrix multiplication" + " "*20, 
        "matrix covariance", 
        "\n" + " "*20 + "linear regression"
        ]
    dataLength = len(operators)             
    angles = np.linspace(0, 2*np.pi, dataLength, endpoint=False)    
    operators.append(operators[0])
    angles = np.append(angles, angles[0])
    monetdb = res_relative["monetdb"].values
    monetdb = np.append(monetdb, monetdb[0])
    pandas = res_relative["pandas"].values
    pandas = np.append(pandas, pandas[0])
    spark = res_relative["spark"].values
    spark = np.append(spark, spark[0])
    pytorch = res_relative["pytorch"].values
    pytorch = np.append(pytorch, pytorch[0])
    tensorra = res_relative["tensorra"].values
    tensorra = np.append(tensorra, tensorra[0])
    plt.polar(angles, monetdb, linestyle="dotted")
    plt.polar(angles, pandas, linestyle="dashed")
    plt.polar(angles, spark, linestyle="dashdot")
    plt.polar(angles, pytorch, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    plt.polar(angles, tensorra)
    plt.thetagrids(angles*180/np.pi, operators)
    plt.legend(labels=("MonetDB", "pandas", "Spark", "PyTorch", "TensorRA"), loc=(1,0), bbox_to_anchor=(1.2,0))
    plt.tick_params("both", pad=2)
    plt.yticks([])
    plt.savefig("motivation.eps", format='eps', bbox_inches = 'tight') 
    plt.show()
    return 0

df = pd.read_csv("pipeline_result.csv")
df_avg = df.groupby(["framework", "pipeline", "datasize"]).apply(clean).reset_index()
df_avg["total"] = df_avg["conversion_time"] + df_avg["linear_time"] + df_avg["relational_time"]
print(df_avg)
#res_operator = pd.read_csv("operator_result.csv")
#res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index()    
#plot_motivation_bar(res_pipeline, res_operator)
#plot_operator(res_operator, "selection")
#plot_operator(res_operator, "projection")
#plot_operator(res_operator, "inner_join")
#plot_operator(res_operator, "groupby")
#plot_operator(res_operator, "aggregation")
