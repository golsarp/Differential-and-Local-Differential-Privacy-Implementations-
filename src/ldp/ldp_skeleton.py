import math, random
import matplotlib.pyplot as plt
import numpy as np

from numpy import random
import numpy as np

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """
def grr_estimator(val,l,eps):
    upper = pow(np.e,eps)
    down = (pow(np.e,eps) + len(DOMAIN)-1)
    p = upper/down
    q = (1-p)/(len(DOMAIN) -1)
    cv = (val - (l*q))/(p-q)
    return cv

def average_error(real_list,perturbed_list):
    real_dict = dict(real_list)
    pert_dict = dict(perturbed_list)
    #print("real dict")
    #print(real_dict)
    keys = list(real_dict.keys())
    error = 0
    bin_l = len(keys)
    for key in keys:
        percent = abs((pert_dict[key] - real_dict[key]))
        error = error + percent
    return error/bin_l






def real_values(dataset):
    real_dict = {}
    for i in range(1,18):
        real_dict[i] = 0
    #print(exp_dict)
    for val in dataset:
        real_dict[val] = real_dict[val] + 1
        
    return list(real_dict.items())



def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


def rappor_estimator(collected_res,epsilon,data_set_len):
    upper = pow(np.e,epsilon/2)
    down = (pow(np.e,epsilon/2) + 1)
    p = upper/down
    q = 1/down
    #print("my p is " + str(p))
    #print("my q is " + str(q))
    #print("upper is "+ str(upper))
    #print("down is "+ str(down))
    
    estimated = (collected_res - (data_set_len*q))/(p-q)
    #print("length " + str(data_set_len))
    #print("collected= " + str(collected_res))
    #print("estimated= " + str(estimated))
    return estimated

def oue_est(collected_res,epsilon,data_set_len):
    #upper = pow(np.e,epsilon/2)
    #down = (pow(np.e,epsilon/2) + 1)
    #p = upper/down
    #q = 1/down
    #print("my p is " + str(p))
    #print("my q is " + str(q))
    #print("upper is "+ str(upper))
    #print("down is "+ str(down))
    
    est = (2*(((pow(np.e,epsilon) + 1)*collected_res) - data_set_len  )) /((pow(np.e,epsilon)) -1 )
    
    #print("length " + str(data_set_len))
    #print("collected= " + str(collected_res))
    #print("estimated= " + str(estimated))
    return est




# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    upper = pow(np.e,epsilon)
    down = (pow(np.e,epsilon) + len(DOMAIN)-1)
    p = upper/down
    domain = DOMAIN.copy()
    index = domain.index(val)
    probs = [0]*len(DOMAIN)
    q = (1-p)/(len(domain) -1)

    for i in range(len(DOMAIN)):
        if i == index:
            #print("in" + str(i))
            probs[i] = p
        else:
            probs[i] = q
    """ 
    print("new dom is ")
    print(domain)
    print("probs are ")
    print(probs)
   
    print("p is " + str(p))
    print("1-p id " + str((1-p)/16))
    print("q is " + str(q))
    print("val is " + str(val))
    """
    #print("probs are ")
    #print(probs)
    #print("sum is ")
    #print(sum(probs))
    
    perturbed = np.random.choice(DOMAIN, 1, p = probs)
    #print("perturbed is ")
    #print(perturbed)
    
    return perturbed[0]
    


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    #print(perturbed_values)
    exp_dict = {}
    for i in range(1,18):
        exp_dict[i] = 0
    #print(exp_dict)
    #return
    for val in perturbed_values:
        exp_dict[val] = exp_dict[val] + 1
        #exp_dict[val] = perturbed_values.count(val)
        
    #print(exp_dict)
    keys = list(exp_dict.keys())
    #print("keys are ")
    #print(keys)
    est_dict = {}
    for i in range(1,18):
        est_dict[i] = 0

    for key in keys:
        reported_val = exp_dict[key]
        est_dict[key] = grr_estimator(reported_val,len(perturbed_values),epsilon)
    #print("estimation is ")
    #print(est_dict)
    res = list(est_dict.items())
    return res
    


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    length = len(dataset)
    perturbed_list = []
    
    for i in range(length):
       perturbed_list.append(perturb_grr(dataset[i], epsilon))
    
    estimation = estimate_grr(perturbed_list, epsilon)
    #print("esti ")
    #print(estimation)
    real = real_values(dataset)
    #print("rea are ")
    #print(real)
    error  = average_error(real,estimation)       
    return error
    


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    domain = DOMAIN.copy()
    index = domain.index(val)
    encoded = [0]* len(DOMAIN)
    encoded[index] = 1
    #print("val is ")
    #print(val)
    #print("encoded is ")
    #print(encoded)
    return encoded
    


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    upper = pow(np.e,epsilon/2)
    down = (pow(np.e,epsilon/2) + 1)
    p = upper/down
    q = 1/down
    #print("q inside perturb " + str(q))
    #print("p inside perturb " + str(p))
    #return

    fl = [1,0]
    probs = [q,p]
    #print("p is " + str(p))
    encoded = encoded_val.copy()
    #print("q is " + str(q))
    #f = np.random.choice(flip, 1, probs)
    #f = f[0]
    #print("f is " + str(f))

    #rand = random.uniform(0, 1)
    #print("rand is " + str(rand))
    for i in range(len(encoded)):
        #r = random.uniform(0, 1)
        f = np.random.choice(fl, 1,
                         p = probs)
        #f = np.random.choice(fl, 1, probs = [1.0,0.0])
        #print("i is " + str(i))
        #print("f is " + str(f))
        f = f[0]
        #print("f is " + str(f))
        #print("f is " + str(f))
        #if f == 1:
        #print("r is ")
        #print(r)
        #if r < q:
        if f == 1:
            #print("flipped")
            #print("in")
            if encoded[i] == 1:
                #print("xxxxx")
                encoded[i] = 0
            else:
                #print("yyyyyyyy")
                #print("en is before  " + str(encoded[i]))
                encoded[i] = 1
                #print("en is after " + str(encoded[i]))
        #print("normal")
        
    #print("flipped ")
    #print(encoded)

    return encoded

    
    


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    # data set is perturbed values 
    collected = [0]*len(DOMAIN)
    
    

    client_num = len(perturbed_values)
    for i in range(client_num):
        client_vec = perturbed_values[i]
        for j in range(len(DOMAIN)):
            collected[j]=  collected[j] + client_vec[j]
    #print("total is ")
    #print(collected)

    estimated = [0]*len(DOMAIN)
    for k in range(len(DOMAIN)):
        estimated[k] = rappor_estimator(collected[k],epsilon,client_num)
        #return
        #print("client num ")
        #print(client_num)
    
    #print("estimated is ")
    #print(estimated)
    #return
    #print("estimaed values are ")
    #print(estimated)
    histogram = []

    for l in range(len(DOMAIN)):
        tup = (DOMAIN[l],estimated[l])
        histogram.append(tup)
    #print("histogram is ")
    #print(histogram)
    return histogram
    pass


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    l = len(dataset)
    clients = []
    for i in range(l):
        cl = encode_rappor(dataset[i])
        clients.append(cl)
    
    #print("clients are ")
    #print(clients)


    perturbe_cl = []
    for i in range(l):
        pert = perturb_rappor(clients[i], epsilon)
        perturbe_cl.append(pert)

    #print("perturbed are  are ")
    #print(perturbe_cl)

    estimated_hist = estimate_rappor(perturbe_cl, epsilon)
    #print("estimation is ")
    #print(estimated_hist)

    true_vals = real_values(dataset)
    #print("true vals are ")
    #print(true_vals)
    error = average_error(true_vals,estimated_hist)
    return error

    pass


# OUE

# TODO: Implement this function!
def encode_oue(val):
    domain = DOMAIN.copy()
    index = domain.index(val)
    encoded = [0]* len(DOMAIN)
    encoded[index] = 1
    #print("val is ")
    #print(val)
    #print("encoded is ")
    #print(encoded)
    return encoded
    


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    upper = pow(np.e,epsilon/2)
    down = (pow(np.e,epsilon/2) + 1)
    p = 1/2
    q = 1/down
    #print("q inside perturb " + str(q))
    #print("p inside perturb " + str(p))
    #return

    fl = [1,0]
    probs = [q,p]
    #print("p is " + str(p))
    encoded = encoded_val.copy()
    #print("q is " + str(q))
    #f = np.random.choice(flip, 1, probs)
    #f = f[0]
    #print("f is " + str(f))

    #rand = random.uniform(0, 1)
    #print("rand is " + str(rand))
    ### new ratios are  here check for the bit to flip
    flip1 = [1,0]
    flip1prob = [1/2,1/2]

    down2 = pow(np.e,epsilon) + 1
    up2 = pow(np.e,epsilon) 
    flip0 = [1,0]
    flip0prob = [1/down2, up2/down2]

    ##FLIP PART İS GOİNG TO CHANGE 
    for i in range(len(encoded)):
       
        #f = np.random.choice(fl, 1,
                         #p = probs)
        #f = np.random.choice(fl, 1, probs = [1.0,0.0])
        #print("i is " + str(i))
        #print("f is " + str(f))
        #f = f[0]
        #print("f is " + str(f))
        #print("f is " + str(f))
        #if f == 1:
        #print("r is ")
        #print(r)

        if encoded[i] == 1:
            f = np.random.choice(fl, 1,p = flip1prob)
            #print("f is " + str(f))
            f = f[0]
            if f == 1:
                encoded[i] = 0
        else:
            f = np.random.choice(fl, 1,p = flip0prob)
            f = f[0]
            if f == 1: 
                encoded[i] = 1
                         
        ## no neeeeddd
        """
        if r < q:
        #if f == 1:
            #print("flipped")
            #print("in")
            if encoded[i] == 1:
                #print("xxxxx")
                encoded[i] = 0
            else:
                #print("yyyyyyyy")
                #print("en is before  " + str(encoded[i]))
                encoded[i] = 1
                #print("en is after " + str(encoded[i]))
        #print("normal")
        """
        
    #print("flipped ")
    #print(encoded)

    return encoded
    pass


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    collected = [0]*len(DOMAIN)
    
    

    client_num = len(perturbed_values)
    for i in range(client_num):
        client_vec = perturbed_values[i]
        for j in range(len(DOMAIN)):
            collected[j]=  collected[j] + client_vec[j]
    #print("total is ")
    #print(collected)

    estimated = [0]*len(DOMAIN)
    for k in range(len(DOMAIN)):
        estimated[k] = oue_est(collected[k],epsilon,client_num)
        #return
        #print("client num ")
        #print(client_num)
    
    #print("estimated is ")
    #print(estimated)
    #return
    #print("estimaed values are ")
    #print(estimated)
    histogram = []

    for l in range(len(DOMAIN)):
        tup = (DOMAIN[l],estimated[l])
        histogram.append(tup)
    #print("histogram is ")
    #print(histogram)
    return histogram
    pass


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    l = len(dataset)
    clients = []
    for i in range(l):
        cl = encode_oue(dataset[i])
        clients.append(cl)
    
    #print("clients are ")
    #print(clients)


    perturbe_cl = []
    for i in range(l):
        pert = perturb_oue(clients[i], epsilon)
        perturbe_cl.append(pert)

    #print("perturbed are  are ")
    #print(perturbe_cl)

    estimated_hist = estimate_oue(perturbe_cl, epsilon)
    #print("estimation is ")
    #print(estimated_hist)

    true_vals = real_values(dataset)
    #print("true vals are ")
    #print(true_vals)
    error = average_error(true_vals,estimated_hist)
    return error

    pass


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")
    #res = perturb_grr(3, 0.1)
    
    #print("res is " + str(res))
    
    #val = estimate_grr([1,5,3,3,5,6,7,8,2,4,5], 6.0)
    #print("val")
   #print(val)
    """
    grr_est = grr_experiment(dataset, 0.1)
    print("Wwwwwwwwwwwwwwwwwwwwwwwwwwww")
    print(grr_est)
    real = real_values(dataset)
    print("reeees")
    print(real)

    error  = average_error(real,grr_est)       
    print("err")
    print(error)
    """
    
    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
        #break

    print("*" * 50)
    
    
    """
    ret = encode_rappor(4)
    pert = perturb_rappor(ret, 4.0)

    ret2 = encode_rappor(7)
    pert2 = perturb_rappor(ret2, 4.0)

    clients = []
    clients.append(pert)
    clients.append(pert2)
    print("clients are ")
    print(clients)
    est = estimate_rappor(clients, 4.0)
    print("estimation is ")
    print(est)
    """
    """
    ret = encode_rappor(4)
    pert = perturb_rappor(ret, 4.0)
    print("original ")
    print(ret)
    print("perturned")
    print(pert)
    rappor_experiment(dataset, 4.0)
    """
   
    
    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)
    
    
    
    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
    

if __name__ == "__main__":
    main()

