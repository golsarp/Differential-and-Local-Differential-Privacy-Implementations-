import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header = 0)
    return df


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    #print("tghis is dataset")
    #print(dataset[chosen_anime_id])
    rt_list = dataset[chosen_anime_id]
    #print("list is ")

   
    rt_dict = {}
    for i in range (-1,11):
        rt_dict[i] = 0
    #print("this is rt_dict")
    #print(rt_dict)
    l = len(rt_list)
    #print("length is " + str(l))
    for i in range(len(rt_list)):
        val = rt_list[i]
        #print("val is")
        #print(val)
        if str(rt_list[i]) != "nan":
            #print("inside")
             rt_dict[rt_list[i]] = rt_dict[rt_list[i]] + 1
        #    rt_dict[-1] = rt_dict[-1] +  1
            
        #else:
             #print("not")
        #rt_dict[rt_list[i]] = rt_dict[rt_list[i]] + 1
        
    #print("this is rt_dict")
    #print(rt_dict)
    keys = list(rt_dict.keys())
    vals = list(rt_dict.values())
    #print("vals are ")
    #print(vals)
    #print("keys are ")
    #print(keys)
    id = dataset[chosen_anime_id]
    print("CLOSE THE HISTOGRAM TO CONTINUE THE EXPERIMENT")
    plt.hist(id,bins=keys)
    #plt.show(block = False)
    plt.title("Rating counts for choosen anime = " + str(chosen_anime_id))
    #plt.draw()
    ## show phistogram for 5 seconds
    #plt.pause(5)
    plt.show()
    
    
    #plt.close()
    new_l = list(rt_dict.items())
    #print("new l ")
    #print(new_l)
    
    return new_l


# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    loc = 0
    cp_counts = dict(counts)
    
    scale = 2/epsilon
    noise =  np.random.laplace(loc, scale)
    #print("noise is ")
    #print(noise)
    #print("counts are ")
    #print(counts[0])
    
    keys = list(cp_counts.keys())
    l = len(keys)
    for key in keys:

        cp_counts[key] = cp_counts[key] + np.random.laplace(loc, scale)
        

    #print("noisy is ")
    dp_list = list(cp_counts.items())
    #print(dp_list)
    return dp_list
    


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    bin_num = len(actual_hist)
    #print("bin num is " + str(bin_num))
    actual_dict = dict(actual_hist)
    #print("actual hist is ")
    #print(actual_dict)
    noisy_dict = dict(noisy_hist)
    #print("noisy hist is ")
    #print(noisy_dict)
    keys = list(actual_dict.keys())
    #print("keys are ")
    #print(keys)
    error = 0
    for key in keys:
        #print("actual" + str(actual_dict[key]))
        #print("noisy " + str(noisy_dict[key]))
        error = error + abs((actual_dict[key] - noisy_dict[key]))
        #print("error is " + str(error))


    #print(actual_hist)
    #print(noisy_hist)
    #print("avg error is ")
    #print(error/bin_num)
    return error/bin_num
    


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    bin_num = len(actual_hist)
    #print("bin num is " + str(bin_num))
    actual_dict = dict(actual_hist)
    #print("actual hist is ")
    #print(actual_dict)
    noisy_dict = dict(noisy_hist)
    #print("noisy hist is ")
    #print(noisy_dict)
    keys = list(actual_dict.keys())
    #print("keys are ")
    #print(keys)
    error = 0
    for key in keys:
        #print("actual" + str(actual_dict[key]))
        #print("noisy " + str(noisy_dict[key]))
        val = (actual_dict[key] - noisy_dict[key])
        error = error + pow(val,2)


    #print(actual_hist)
    #print(noisy_hist)
    #print("mean sq error  is ")
    #print(error)
    return error/bin_num


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    avg_error_stat = []
    mse_error_stat = []

    avg_error_list = []
    mse_error_list = []
    for i in range(len(eps_values)):
        eps = eps_values[i]
        for i in range(41):
            dp_counts = get_dp_histogram(counts,eps)
            avg_error = calculate_average_error(counts,dp_counts)
            mse_error = calculate_mean_squared_error(counts,dp_counts)
            avg_error_list.append(avg_error)
            mse_error_list.append(mse_error)

        
        val1 = sum(avg_error_list)/len(avg_error_list)
        avg_error_stat.append(val1)    

        val2 = sum(mse_error_list)/len(mse_error_list)
        mse_error_stat.append(val2)
        avg_error_list =[]
        mse_error_list = []
        


    return (avg_error_stat ,mse_error_stat)


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    all_keys = list(dataset.keys())
    #print(type(dataset))

    #return
    anime_dict = {}
    
    i = 0
    for key in all_keys:
        if str(key) != "user_id":
            #print("vehh")
            #print(key)
            anime_dict[key] = 0

    keys = list(anime_dict.keys())
    #print("keys are ")
    #print(keys)
    #print("len is ")
    
    for key in keys:
    #print(len(dataset))
        #print("bah ahb")
        #print(dataset[key])
        #print("deneme")
        #break
        for i in range(len(dataset)):
            if dataset[key][i] == 10:
                anime_dict[key] =  anime_dict[key] + 1
            #print(dataset[key][2])
        #break
    
    pr_list = {}
    for key in keys:
        pr_list[key] = 0
    
    for key in keys:
        pr_list[key] = pow(np.e,(epsilon*anime_dict[key]/1*2))
    
    #print("last ")
    #print(anime_dict)
    #print("e epsilon acl")
    #print(pr_list)
    total = sum(list(pr_list.values()))
    #print("total " + str(total))
    
    #prob_list = pr_list.copy()
    #for key in keys:
    #    prob_list[key] = prob_list[key]/total
    #print("probs are ")
    #print(prob_list)
    
    val_list = list(pr_list.keys())
    weight_list = list(pr_list.values())
    
    #print("val list")
    #print(val_list)
    #print("weights")
    #print(weight_list)
    #print("choosen one ")
    #choosen = np.random.choice(val_list, 1, p = prob_list)
    choosen = random.choices(val_list, weight_list, k=1)
    #print(choosen)

    
    return choosen[0]






def expo_result(anime_dict,epsilon):
    keys = list(anime_dict.keys())
    pr_list = {}
    for key in keys:
        pr_list[key] = 0
    #print("anime dict")
    #print(anime_dict)
    # here sensivity is 1 
    for key in keys:
        pr_list[key] = pow(np.e,(epsilon*anime_dict[key]/(1*2)))
    
    #print("last ")
    #print(anime_dict)
    #print("e epsilon acl")
    #print(pr_list)
    total = sum(list(pr_list.values()))
    #print("total num " + str(total))
    #print("pr " )
    #print(pr_list)
    #return
    
    prob_list = pr_list.copy()
    for key in keys:
        prob_list[key] = prob_list[key]/total
    #print("probs are ")
    #print(prob_list)
    total = sum(list(prob_list.values()))
    #print("total " + str(total))
    #print("prob list is ")
    #print(prob_list)
    #return 
    val_list = list(pr_list.keys())
    weight_list = list(pr_list.values())
    
    #print("val list")
    #print(val_list)
    #print("weights")
    #print(weight_list)
    #print("choosen one ")

    choosen = np.random.choice(val_list, 1, p = list(prob_list.values()))
    #choosen = random.choices(val_list, weight_list, k=1)
    #print(choosen)

    
    return choosen[0]






# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    all_keys = list(dataset.keys())
    #print(dataset[keys[1]])
    anime_dict = {}
    
    i = 0
    for key in all_keys:
        if str(key) != "user_id":
            #print("vehh")
            #print(key)
            anime_dict[key] = 0

    keys = list(anime_dict.keys())
    #print("keys are ")
    #print(keys)
    #print("len is ")
    for key in keys:
    #print(len(dataset))
        #print("bah ahb")
        #print(dataset[key])
        #print("deneme")
        #break
        for i in range(len(dataset)):
            if dataset[key][i] == 10:
                anime_dict[key] =  anime_dict[key] + 1
    #print("dict is ")
    #print(anime_dict)
    
    k = list(anime_dict.keys())
    v = list(anime_dict.values())
    max_val = v.index(max(v))
    true_val = k[max_val]
    result = []
    count = 0
    #print("true val is ")
    #print(true_val)
    #print("anime dict is ")
    #print(anime_dict)

    

    loop = 1000
    for i in range(len(eps_values)):
        eps = eps_values[i]
        #print("ssssss")
        #print("is is " + str(i))
        for j in range(loop):
            #print("xxxxxx")
            #expo_res = most_10rated_exponential(dataset,eps)
            expo_res = expo_result(anime_dict,eps)
            #print("expo res is " + str(expo_res))
            if str(expo_res) == str(true_val):
                #print("got exeeeee")
                count = count + 1
        percent = (count/loop)*100
        #print("percent res is " + str(percent))
        result.append(percent)
        count = 0

    #print("result ")
    #print(result)
    return result
    


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)
    counts = get_histogram(dataset)
    """
    counts = get_histogram(dataset)
    dp_counts = get_dp_histogram(counts,0.001)
    av_error = calculate_average_error(counts,dp_counts)
    ex_error = calculate_mean_squared_error(counts,dp_counts)
    #print(dp_counts)
    """
    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])
    
    """
    choosen = most_10rated_exponential(dataset,0.001)
    #print("choosen val is ")
    #print(choosen)
    """
    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])
    

if __name__ == "__main__":
    main()

