"""
Plot the experiment results
"""
from re import I
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

model_list = [
        "Binarizer",
        "Normalizer",
        "MinMaxScaler",
        "RobustScaler",
        "LinearRegression",
        "LogisticRegression",
        "SGDClassifier",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "RandomForestClassifier",
        "LinearSVC",
        "LinearSVR"
        ]

model_simplified = [
        "Binarizer",
        "Normalizer",
        "MinMaxScaler",
        "RobustScaler",
        "LinearReg",
        "LogisticReg",
        "SGDClassifier",        
        "DecTreeClf",
        "DecTreeReg",
        "RandForestClf",
        "LinearSVC",
        "LinearSVR"
        ]

def compare_and_bold(*target_list):
    """
    Make comparsion between data and set the smallest as bold
    return them in latex format
    """
    is_min_list = []
    str_list = []
    for i in target_list:
        is_min = True
        for j in target_list:
            if(i > j and j!=0):
                is_min = False
        if(i == 0):
            is_min = "Nan"
        is_min_list.append(is_min)
    for i in range(len(target_list)):
        if(is_min_list[i] == "Nan"):
            str_list.append("-")
        elif(is_min_list[i] == True):
            str_list.append("\\textbf{" + str(target_list[i]) + "}")
        else:
            str_list.append(str(target_list[i]))
    return str_list

def read_rasp_framework(filename):
    """
    Read rasp framework, fill nan with 0
    """
    df_rasp = pd.read_csv(filename)
    
    dataset_list = df_rasp["dataset"].unique().tolist()
    for model in model_list:
        tmp = df_rasp[df_rasp["model"]==model]
        if(tmp.empty):
            for dataset in dataset_list:
                df_rasp = df_rasp.append({'model' : model , 'dataset' : dataset, 'sklearn' : 0, 'llvm' : 0} , ignore_index=True)
    df_avg = df_rasp.groupby(["model", "dataset"]).mean().reset_index()
    df_avg[["sklearn", "ucab"]] = df_avg[["sklearn", "llvm"]] * 1000
    df_rasp = df_avg[["model", "dataset", "sklearn", "ucab"]]
    return df_rasp

def read_framework(filename):
    """
    Read CPU and GPU data
    """
    df = pd.read_csv(filename) 
    df_avg = df.groupby(["model", "dataset"]).mean().reset_index()
    df_avg["hb_cpu"] = df_avg[["hb_torch_cpu", "hb_tvm_cpu"]].min(axis=1)
    df_avg["hb_gpu"] = df_avg[["hb_torch_gpu", "hb_tvm_gpu"]].min(axis=1)
    df_avg["ucab_cpu"] = df_avg[["llvm", "llvm -mcpu=core-avx2"]].min(axis=1)
    df = df_avg[["model", "dataset", "sklearn", "hb_cpu", "ucab_cpu", "hb_gpu", "cuda"]]
    df[["sklearn", "hb_cpu", "ucab_cpu", "hb_gpu", "cuda"]] = df[["sklearn", "hb_cpu", "ucab_cpu", "hb_gpu", "cuda"]] * 1000
    # Set RandomForestClassifier cuda to 0
    index = df[df["model"] == "RandomForestClassifier"].index.tolist()[0]
    df.iloc[index, -1] = 0
    return df

def read_optimization(opt_file, elim_file):
    """
    read file and return dataframe
    opt_file is the optimization file with sparse replacing and dtype rewriting
    elim_file is the optimization file with redundant elimination
    """
    df_optimization = pd.read_csv(opt_file)
    df_elimination = pd.read_csv(elim_file)
    df_opt_avg = df_optimization.groupby(["model", "dataset"]).mean().reset_index() 
    df_opt_avg[["base", "dtype", "sparse", "both"]] = df_opt_avg[["base", "dtype", "sparse", "both"]] * 1000
    df_elim_avg = df_elimination.groupby(["model", "dataset"]).mean().reset_index() 
    df_elim_avg[["base", "elimination"]] = df_elim_avg[["base", "elimination"]] * 1000
    return df_opt_avg, df_elim_avg

def complete_rasp_data(opt, elim, framework):
    opt_models = opt["model"].unique().tolist()
    tmp = framework[framework["model"].isin(opt_models)]["sklearn"]
    opt["sklearn"] = framework[framework["model"].isin(opt_models)]["sklearn"].values
    for model in opt_models:
        index = framework[framework["model"] == model].index.tolist()[0]
        framework_value = framework[framework["model"] == model]["ucab"].values[0]
        opt_value = opt[opt["model"] == model]["both"].values[0]
        if(framework_value <= 0 or framework_value > opt_value):    
            framework.iloc[index, 3] = opt_value
    elim_models = elim["model"].unique().tolist()
    for model in elim_models:
        index = framework[framework["model"] == model].index.tolist()[0]
        elim_value = elim[elim["model"] == model]["elimination"].values[0]
        framework.iloc[index, 3] = elim_value
    tmp = framework[framework["model"].isin(elim_models)]["sklearn"].values
    elim["sklearn"] = tmp
    return opt, elim, framework

def complete_data(opt, elim, framework):
    opt_models = opt["model"].unique().tolist()
    tmp = framework[framework["model"].isin(opt_models)]["sklearn"]
    opt["sklearn"] = framework[framework["model"].isin(opt_models)]["sklearn"].values
    for model in opt_models:
        index = framework[framework["model"] == model].index.tolist()[0]
        framework_value = framework[framework["model"] == model]["ucab_cpu"].values[0]
        opt_value = opt[opt["model"] == model]["both"].values[0]
        if(framework_value <= 0 or framework_value > opt_value):    
            framework.iloc[index, 4] = opt_value
    elim_models = elim["model"].unique().tolist()
    for model in elim_models:
        index = framework[framework["model"] == model].index.tolist()[0]
        elim_value = elim[elim["model"] == model]["elimination"].values[0]
        framework.iloc[index, 4] = elim_value    
    tmp = framework[framework["model"].isin(elim_models)]["sklearn"].values
    elim["sklearn"] = tmp
    return opt, elim, framework

def read_breakdown():
    df = pd.read_csv("breakdown.csv")
    df_avg = df.groupby(["model", "dataset", "target"]).mean().reset_index()
    columns = ["model", "dataset", "target", "IO", "computation"]
    df_new = pd.DataFrame(columns=columns)
    for model in model_list:
        df_new = df_new.append(df_avg[df_avg["model"]==model])
    df_avg = df_new
    df_avg["IO"] = df_avg["IO"] * 1000
    df_avg["computation"] = df_avg["computation"] * 1000
    df_avg["sum"] = df_avg["IO"] + df_avg["computation"]
    df_avg["IO_ratio"] = df_avg["IO"] / df_avg["sum"] * 100
    df_avg["computation_ratio"] = df_avg["computation"] / df_avg["sum"] * 100
    gpu_breakdown = df_avg[df_avg["target"]=="cuda"][["model", "IO", "computation", "sum", "IO_ratio", "computation_ratio"]]
    # Set RandomForestClassifier cuda to 0
    gpu_breakdown.iloc[9, 1:] = 0    
    cpu_breakdown = df_avg[df_avg["target"]=="llvm"][["model", "IO", "computation", "sum", "IO_ratio", "computation_ratio"]]
    df = pd.read_csv("rasp_breakdown.csv")
    df_avg = df.groupby(["model", "dataset"]).mean().reset_index()
    columns = ["model", "dataset", "target", "IO", "computation"]
    df_new = pd.DataFrame(columns=columns)
    for model in model_list:
        df_new = df_new.append(df_avg[df_avg["model"]==model])   
    df_avg = df_new
    df_avg["IO"] = df_avg["IO"] * 1000
    df_avg["computation"] = df_avg["computation"] * 1000
    df_avg["sum"] = df_avg["IO"] + df_avg["computation"]
    df_avg["IO_ratio"] = df_avg["IO"] / df_avg["sum"] * 100
    df_avg["computation_ratio"] = df_avg["computation"] / df_avg["sum"] * 100
    iot_breakdown = df_avg[["model", "IO", "computation", "sum", "IO_ratio", "computation_ratio"]]
    return gpu_breakdown, cpu_breakdown, iot_breakdown

def plot_framework(df, df_rasp):
    """
    The result of framework comparsion, in a latex table format
    """ 
    df["speedup_cpu_sklearn"] = df["sklearn"] / df["ucab_cpu"]
    df["speedup_cpu_hb"] = df["hb_cpu"] / df["ucab_cpu"]
    df["speedup_gpu_hb"] = df["hb_gpu"] / df["cuda"]
    print(df)
    df_rasp["speedup"] = df_rasp["sklearn"] / df_rasp["ucab"]
    print(df_rasp)
    for model in model_list:
        index = df[df["model"]==model].index.tolist()[0]
        sklearn = int(df.loc[index, "sklearn"])
        hb_cpu = int(df.loc[index, "hb_cpu"])
        ucab_cpu = int(df.loc[index, "ucab_cpu"])
        hb_gpu = int(df.loc[index, "hb_gpu"])
        ucab_gpu = int(df.loc[index, "cuda"])
        sklearn_iot = int(df_rasp.loc[index, "sklearn"])
        ucab_iot = int(df_rasp.loc[index, "ucab"])     
        cpu_res = compare_and_bold(sklearn, hb_cpu, ucab_cpu)
        gpu_res = compare_and_bold(hb_gpu, ucab_gpu)
        iot_res = compare_and_bold(sklearn_iot, ucab_iot)
        print(
                model + " & " + 
                cpu_res[0] + " & " + 
                cpu_res[1] + " & " +
                cpu_res[2] + " & " +
                gpu_res[0] + " & " + 
                gpu_res[1] + " & " +
                iot_res[0] + " & " +
                iot_res[1] +
                "\\\\"
              )

def plot_breakdown(gpu_breakdown, cpu_breakdown, iot_breakdown):
    """
    The result of breakdown
    """
    #print(gpu_breakdown)
    #print(cpu_breakdown)
    #print(iot_breakdown)
    #print(iot_breakdown["IO_ratio"].mean()*12/11)
    #print(cpu_breakdown["computation"] / gpu_breakdown["computation"])
    #print(np.array(cpu_breakdown["computation"].tolist()) / np.array(gpu_breakdown["computation"].tolist()))
    width = 0.6
    index = np.arange(len(model_list))
    figure, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)   
    gpu_IO = ax1.bar(index, gpu_breakdown["IO_ratio"], width)
    gpu_computation = ax1.bar(index, gpu_breakdown["computation_ratio"], width, bottom=gpu_breakdown["IO_ratio"])    
    ax1.legend((gpu_IO, gpu_computation), 
               ("IO", "computation"),loc = 3, fontsize=12)
    ax1.set_ylabel('percentage', fontsize=12)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(labelsize=12)
    ax1.set_title("GPU breakdown", fontsize=14)
    cpu_IO = ax2.bar(index, cpu_breakdown["IO_ratio"], width)
    cpu_computation = ax2.bar(index, cpu_breakdown["computation_ratio"], width, bottom=cpu_breakdown["IO_ratio"])    
    ax2.legend((cpu_IO, cpu_computation), 
               ("IO", "computation"),loc = 3, fontsize=12)
    ax2.set_ylabel('percentage', fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_title("CPU breakdown", fontsize=14)
    ax2.tick_params(labelsize=12)
    iot_IO = ax3.bar(index, iot_breakdown["IO_ratio"], width)
    iot_computation = ax3.bar(index, iot_breakdown["computation_ratio"], width, bottom=iot_breakdown["IO_ratio"])    
    ax3.legend((iot_IO, iot_computation), 
               ("IO", "computation"),loc = 3, fontsize=12)
    ax3.set_ylabel('percentage', fontsize=12)  
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_title("IOT breakdown", fontsize=14)
    ax3.tick_params(labelsize=12)
    plt.xticks(index, model_simplified, rotation=90, fontsize=12)
    plt.xlabel("model", fontsize=16)
    figure.subplots_adjust(hspace=0.2) 
    plt.savefig("breakdown.eps", format='eps', bbox_inches = 'tight') 

def plot_optimization(opt, elim, savefile):
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    width = 0.6
    color=["tab:cyan", "tab:blue", "tab:orange", "tab:red"]
    decision_tree = opt.iloc[0][["sklearn", "base", "dtype", "both"]].tolist()
    print(decision_tree)
    print(decision_tree[0]/decision_tree[1])
    print(decision_tree[1]/decision_tree[2])
    print(decision_tree[2]/decision_tree[3])
    print(decision_tree[1]/decision_tree[3])
    index = [0, 1, 2, 3]
    opt_list = ["sklearn", "base", "+dtype", "+sparse"]
    ax1.bar(index, decision_tree, width, color=color) 
    ax1.set_ylabel('time(ms)', fontsize=12)
    ax1.set_xticks(index)
    ax1.set_xticklabels(opt_list, fontsize=10)
    ax1.set_xlabel('DecisionTreeClassifier', fontsize=12)
    
    random_forest = opt.iloc[2][["sklearn", "base", "dtype", "both"]].tolist()
    print(random_forest)
    print(random_forest[0]/random_forest[1])
    print(random_forest[1]/random_forest[2])
    print(random_forest[2]/random_forest[3])
    print(random_forest[1]/random_forest[3])
    index = [0, 1, 2, 3]
    opt_list = ["sklearn", "base", "+dtype", "+sparse"]
    ax2.bar(index, random_forest, width, color=color) 
    ax2.set_ylabel('time(ms)', fontsize=12)
    ax2.set_xticks(index)
    ax2.set_xticklabels(opt_list, fontsize=10)
    ax2.set_xlabel('RandomForestClassifier', fontsize=12)
    
    color=["tab:cyan", "tab:blue", "tab:pink"]
    sgd_classifier = elim.iloc[0][["sklearn", "base", "elimination"]].tolist()
    print(sgd_classifier)
    print(sgd_classifier[0]/sgd_classifier[1])
    print(sgd_classifier[1]/sgd_classifier[2])
    print(sgd_classifier[0]/sgd_classifier[2])
    index = [0, 1, 2]
    opt_list = ["sklearn", "base", "+elimination"]
    ax3.bar(index, sgd_classifier, width, color=color) 
    ax3.set_ylabel('time(ms)', fontsize=12)
    ax3.set_xticks(index)
    ax3.set_xticklabels(opt_list, fontsize=10)
    ax3.set_xlabel('SGDClassifier', fontsize=12) 
    
    figure.subplots_adjust(wspace=0.4)
    plt.savefig(savefile + ".eps", format='eps', bbox_inches = 'tight') 

def plot_mix(filename, savefile):
    df = pd.read_csv(filename)
    df_avg = df.groupby(["framework"]).mean().reset_index()
    df_avg["time"] = df_avg["time"] * 1000
    
    baseline_cpu = df_avg.iloc[0, 1]
    ucab_cpu = df_avg.iloc[1, 1]
    ucab_iot = df_avg.iloc[2, 1]
    print(baseline_cpu/ucab_cpu)
    print(ucab_iot)
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=False)
    width = 0.6
    color=["tab:blue", "tab:orange"]
    
    index = [0, 1]
    cpu_res = [baseline_cpu, ucab_cpu]
    framework = ["baseline", "ucml"]
    ax1.bar(index, cpu_res, width, color=color) 
    ax1.set_ylabel('latency(ms)', fontsize=12)
    ax1.set_xticks(index)
    ax1.set_xticklabels(framework, fontsize=12)
    ax1.set_xlabel('CPU', fontsize=16)
    
    iot_res = [0, ucab_iot]
    ax2.bar(index, iot_res, width, color=color) 
    ax2.set_ylabel('latency(ms)', fontsize=12)
    ax2.set_xticks(index)
    ax2.set_xticklabels(framework, fontsize=12)
    ax2.set_xlabel('Raspberrypi', fontsize=16)
    
    figure.subplots_adjust(wspace=0.4)
    plt.savefig(savefile + ".eps", format='eps', bbox_inches = 'tight') 

def plot_motivation(rasp_data, cpu_gpu_data, model_name):
    """
    Plot the result of comparsion between sklearn and our work as a motivation
    Take RobustScaler and DecisionTreeClassifier as example
    """    
    hardware_list = ["CPU", "GPU", "IoT"]
    sklearn_cpu = float(cpu_gpu_data[cpu_gpu_data["model"] == model_name]["sklearn"])
    sklearn_cpu = float(cpu_gpu_data[cpu_gpu_data["model"] == model_name]["sklearn"])
    sklearn_gpu = 0
    sklearn_rasp = float(rasp_data[rasp_data["model"] == model_name]["sklearn"])
    sklearn_data = [sklearn_cpu, sklearn_gpu, sklearn_rasp]
    ucml_cpu = float(cpu_gpu_data[cpu_gpu_data["model"] == model_name]["ucab_cpu"])
    ucml_gpu = float(cpu_gpu_data[cpu_gpu_data["model"] == model_name]["cuda"])
    ucml_rasp = float(rasp_data[rasp_data["model"] == model_name]["ucab"])
    ucml_data = [ucml_cpu, ucml_gpu, ucml_rasp]
    width = 0.3
    index = np.arange(len(hardware_list))
    plt.figure(figsize=(4,3))
    plt.bar(index-width/2, sklearn_data, width, label="sklearn")
    plt.bar(index+width/2, ucml_data, width, label="our work")
    plt.yscale("log")
    plt.ylabel("time(ms)")
    plt.legend()
    plt.xticks(index, hardware_list)
    plt.xlabel(model_name)
    savefile = "motivation_" + model_name
    plt.savefig(savefile + ".eps", format='eps', bbox_inches = 'tight') 

rasp_framework = read_rasp_framework("rasp_framework.csv")
rasp_opt, rasp_elim = read_optimization("rasp_optimization.csv","rasp_elimination.csv")
rasp_opt, rasp_elim, rasp_framework = complete_rasp_data(rasp_opt, rasp_elim, rasp_framework)
framework = read_framework("framework.csv")
opt, elim = read_optimization("optimization.csv","elimination.csv")
opt, elim, framework = complete_data(opt, elim, framework)
gpu_breakdown, cpu_breakdown, iot_breakdown = read_breakdown()

#print(framework)
"""
plot_framework(framework, rasp_framework)
plot_breakdown(gpu_breakdown, cpu_breakdown, iot_breakdown)
plot_optimization(opt, elim, "CPU_optimization")
plot_optimization(rasp_opt, rasp_elim, "IOT_optimization")
plot_mix("mix.csv", "mix")
"""
plot_motivation(rasp_framework, framework, "RobustScaler")
plot_motivation(rasp_framework, framework, "DecisionTreeClassifier")
plt.show()