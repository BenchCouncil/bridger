import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean,gstd
from matplotlib.patches import Ellipse

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
    ax.bar(x - width, df_pandas, width, label="pandas")
    ax.bar(x, df_spark, width, label="spark")
    ax.bar(x + width, df_tensorra, width, label="tensorra")
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
    plt.xticks(ind, ("pandas", "spark", "tensorra"))
    plt.yscale("log")
    conversion = df["conversion_time"].values
    linear = df["linear_time"].values
    relational = df["relational_time"].values
    width = 0.25
    plt.bar([0.75,1.75,2.75], conversion, width=width, color="c", label="conversion")
    plt.bar([1, 2, 3], linear, width=width, hatch="/", label="linear")
    plt.bar([1.25, 2.25, 3.25], relational, width=width, color="g", hatch=".", label="relational")
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
    #df_cov = df_avg[df_avg["pipeline"] == "covariance"]
    plot_linear_regression(df_lr)
    #plot_conference_covariance(df_cov)
    #plot_relative_performance("cov_relative", df_cov)
    plot_relative_performance("lr_relative", df_lr)
    return 0

def g_mean(x):
    #calculate geometric mean 
    tmp = np.log(x)
    return np.exp(tmp.mean())

def g_std(x):
    #calculate geometric standard deviation
    x_g_mean = g_mean(x)
    x_sum = np.sum()
    return np.exp()

def plot_ellipse(ax, res_pipeline, res_operator, name):
    if(name == "monetdb"):
        center_x = 0
        relational = res_operator[res_operator["framework"] == name]["exec_time"].reset_index()["exec_time"]
        relational_tensorra = res_operator[res_operator["framework"] == "tensorra"]["exec_time"].reset_index()["exec_time"]
        relational_relative = relational_tensorra / relational
        #width = gstd(linear_relative.values)
        #center_y = gmean(relational_relative.values)
        center_y = gmean(relational_relative)
        #height = gmean(relational_relative.values)
    elif(name == "pytorch"):
        center_x = 1
        center_y = 0
    else:
        linear = res_pipeline[res_pipeline["framework"] == name]["linear_time"].reset_index()["linear_time"]
        relational = res_operator[res_operator["framework"] == name]["exec_time"].reset_index()["exec_time"]
        linear_tensorra = res_pipeline[res_pipeline["framework"] == "tensorra"]["linear_time"].reset_index()["linear_time"]
        relational_tensorra = res_operator[res_operator["framework"] == "tensorra"]["exec_time"].reset_index()["exec_time"]
        print(linear)
        print(linear_tensorra)
        print(relational)
        print(relational_tensorra)
        linear_relative = linear_tensorra / linear
        relational_relative = relational_tensorra / relational
        print(linear_relative)
        print(relational_relative)
        #center_x = gmean(linear_relative.values)
        center_x = gmean(linear_relative)
        #width = gstd(linear_relative.values)
        #center_y = gmean(relational_relative.values)
        center_y = gmean(relational_relative)
        #height = gmean(relational_relative.values)
    width = 0.4
    height = 0.2
    ellipse = Ellipse(xy = (center_x, center_y), width = width, height = height, facecolor= 'yellow')
    ax.text(center_x, center_y, name, horizontalalignment="center", multialignment="center")
    ax.add_patch(ellipse)
    return ax

def plot_motivation(res_pipeline, res_operator):
    res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index()    
    res_pipeline = res_pipeline.groupby(["framework", "pipeline", "datasize"]).apply(clean).reset_index()
    #Choose the result of medium size
    res_operator = res_operator[res_operator["row_num"] == 100000]
    res_pipeline = res_pipeline[res_pipeline["datasize"] == 10000]
    
    print(res_operator)
    print(res_pipeline)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax = plot_ellipse(ax, res_pipeline, res_operator, "spark")
    ax = plot_ellipse(ax, res_pipeline, res_operator, "tensorra")
    ax = plot_ellipse(ax, res_pipeline, res_operator, "pandas")
    ax = plot_ellipse(ax, res_pipeline, res_operator, "monetdb")
    ax = plot_ellipse(ax, res_pipeline, res_operator, "pytorch")
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Linear Algebra')
    ax.set_ylabel('Relational Algebra') 
    plt.savefig("motivation.eps", format='eps', bbox_inches = 'tight') 
    plt.show()
    #df_avg = res_pipeline.groupby(["framework", "pipeline", "datasize"]).apply(clean).reset_index()
    return 0

res_pipeline = pd.read_csv("pipeline_result.csv")
res_operator = pd.read_csv("operator_result.csv")
res_operator = res_operator.groupby(["framework", "operator", "row_num"]).apply(clean).reset_index()    
print(res_operator)
#plot_motivation(res_pipeline, res_operator)
plot_operator(res_operator, "selection")
plot_operator(res_operator, "projection")
plot_operator(res_operator, "inner_join")
plot_operator(res_operator, "groupby")
plot_operator(res_operator, "aggregation")
#plot_framework("pipeline_result.csv")