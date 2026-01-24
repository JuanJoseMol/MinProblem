import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib


fm.fontManager.addfont('../../../dataset/Helvetica Neue Bold.ttf')
matplotlib.rc('font', family='Helvetica Neue')

# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue']})
# ## for Palatino and other serif fonts use:
# #matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rc('text', usetex=True)


#plt.rc('font', family='sans-serif')
#plt.rcParams['font.family'] = u'Helvetica Neue'


plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] ='bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18



# This function reads the files and returns a list of arrays

def bestTimeGau(tipo2):
    print ("best gaussian")
    v = np.pi
    if tipo2 == "human":
        #l = [0.0001,0.005,.01,0.05,0.1,0.5,1,1.5,2,3,4,5,6,7,8]
        l = [5*v, 10*v,15*v, 20*v,30*v]#[5*v,10*v,15*v,20*v,25*v,30*v,35*v,40*v,45*v,50*v]
        n_experiments = 44
        
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                #path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.001-w235.61944901923448a0.012-'+str(k) 
                path = 'FBGau'+tipo2+'lr0.01-w'+str(i)+'a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
            
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
    if tipo2 == "brain":
        #l = [i for i in range(1,33)]
        l = [5*v, 10*v,15*v, 20*v,30*v]#[5*v,10*v,15*v,20*v,23.74*v,30*v,35*v,40*v,45*v,50*v,60*v,70*v,80*v,90*v]
        n_experiments = 43
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                #path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.001-w282.7433388230814a0.012-'+str(k)
                path = 'FBGau'+tipo2+'lr0.01-w'+str(i)+'a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                    temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
    if tipo2 == "3d":
        #l = [i for i in range(1,40)]
        l = [1*v, 2*v, 3*v, 4*v, 5*v, 7*v,8*v]#[1*v, 2*v, 3*v, 4*v, 5*v, 6*v, 7*v, 8*v, 9*v, 10*v]
        n_experiments = 43
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                #path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) 
                path = 'FBGau'+tipo2+'lr0.01-w'+str(i)+'a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]



def bestTimeFix(tipo2):
    print ("best design fixed")
    if tipo2 == "human":
        l = [0.0001,0.005,.01,0.05,0.1,0.5,1,1.5,2,3,4,5,6,7,8]
        #l = [i for i in range(1,38)]
        n_experiments = 44
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w235.61944901923448a0.012-'+str(k) 
                #path = str(i)+'RhoMean'+tipo2+'lr0.001-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
        best_mean_function = min(mean_values, key=lambda x: x[2])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[2])
        best_min_function = min(min_values, key=lambda x: x[2])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[2])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[2])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[2])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
    if tipo2 == "brain":
        #l = [i for i in range(1,33)]
        l = [i for i in range(1,30)]
        n_experiments = 43
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w282.7433388230814a0.012-'+str(k)
                #path = str(i)+'RhoMean'+tipo2+'lr0.001-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
    if tipo2 == "3d":
        l = [i for i in range(1,30)]
        #l = [i for i in range(1,83)]
        n_experiments = 35
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) 
                #path = str(i)+'RhoMean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]




def plotFix(tipo):
    tipo2 = "human"
    mean, median, minimum = bestTimeFix(tipo2)
    meanGau, medianGau, minimumGau = bestTimeGau(tipo2)
    if tipo == "mean" and tipo2 == "human":
        l = [mean]
        lGau = [meanGau]
    if tipo == "median" and tipo2 == "human":
        l = [median]
        lGau = [medianGau]
    if tipo == "min" and tipo2 == "human":
        l = [minimum]
        lGau = [minimumGau]
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []
    for i, G in zip(l,lGau):
        for k in range(n_experiments):
            path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            values.append(array)
            path = 'Uni'+tipo2+'lr0.01-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesuni.append(array)
            path = 'FBGau'+tipo2+'lr0.01-w'+str(G)+'a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesnorm.append(array)
            
            
            # Store results
    applied_results = np.array(values)  
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)

    fig, axs = plt.subplots(1, 3, figsize =(13, 4))

    # Fill the region between min and max values
    axs[1].fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    axs[1].fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    axs[1].fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1].plot(median_values, color='blue', linewidth=2, label='RhoMean')
    axs[1].plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    axs[1].plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1].set_xlabel('iterations')
    #axs[1].set_ylabel('Loss')
    axs[1].set_yscale("log")
    axs[1].set_title(tipo2)
    axs[1].set_yticks([1e0,1e-4])
    axs[1].set_xticks([0,1000])
    axs[1].xaxis.set_label_coords(0.5, -0.05)
    #axs[1].legend()
    
    ##################################################################################################

    tipo2 = "brain"
    mean, median, minimum = bestTimeFix(tipo2)
    meanGau, medianGau, minimumGau = bestTimeGau(tipo2)
    if tipo == "mean" and tipo2 == "brain":
        l = [mean]
        lGau = [meanGau]
    if tipo == "median" and tipo2 == "brain":
        l = [median]
        lGau = [medianGau]
    if tipo == "min" and tipo2 == "brain":
        l = [minimum]
        lGau = [minimumGau]
    #l = [12]
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []
    for i, G in zip(l,lGau):
        for k in range(n_experiments):
            path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            values.append(array)
            path = 'Uni'+tipo2+'lr0.01-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesuni.append(array)
            path = 'FBGau'+tipo2+'lr0.01-w'+str(G)+'a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesnorm.append(array)
            
            # Store results
    applied_results = np.array(values)  
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)



    # Fill the region between min and max values
    axs[0].fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    axs[0].fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    axs[0].fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[0].plot(median_values, color='blue', linewidth=2, label='RhoMean')
    axs[0].plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    axs[0].plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('Error')
    axs[0].set_yscale("log")
    axs[0].set_title(tipo2)
    axs[0].set_yticks([1e0,1e-4])
    axs[0].set_xticks([0,1000])
    axs[0].xaxis.set_label_coords(0.5, -0.05)
    axs[0].yaxis.set_label_coords(-0.05, 0.5)
    #axs[0].legend()

    ##################################################################################################


    tipo2 = "3d"
    mean, median, minimum, = bestTimeFix(tipo2)
    meanGau, medianGau, minimumGau = bestTimeGau(tipo2)
    if tipo == "mean" and tipo2 == "3d":
        l = [mean]
        lGau = [meanGau]
    if tipo == "median" and tipo2 == "3d":
        l = [median]
        lGau = [medianGau]
    if tipo == "min" and tipo2 == "3d":
        l = [minimum]
        lGau = [minimumGau]
    #l = [27]
    n_experiments = 35
    values = []
    valuesuni = []
    valuesnorm = []
    for i, G in zip(l,lGau):
        for k in range(n_experiments):
            path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            values.append(array)
            path = 'Uni'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesuni.append(array)
            path = 'FBGau'+tipo2+'lr0.01-w'+str(G)+'a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesnorm.append(array)
            
            # Store results
    applied_results = np.array(values)  
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)


    # Fill the region between min and max values
    axs[2].fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    axs[2].fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    axs[2].fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[2].plot(median_values, color='blue', linewidth=2, label='Design')
    axs[2].plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    axs[2].plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[2].set_xlabel('iterations')
    #axs[2].set_ylabel('Loss')
    axs[2].set_yscale("log")
    axs[2].set_title(tipo2)
    axs[2].set_yticks([1e0,1e-3])
    axs[2].set_xticks([0,1000])
    axs[2].legend(frameon=False, loc='upper right', fontsize='small')
    axs[2].xaxis.set_label_coords(0.5, -0.05)
    plt.savefig('speedres/N'+tipo+'Meanfinal.pdf')
    plt.show()

    
 
  
 
    return None


def bestTime(tipo2):
    print ("best design")
    if tipo2 == "human":
        #l = [0.0001,0.005,.01,0.05,0.1,0.5,1,1.5,2,3,4,5,6,7,8]
        l = [i for i in range(1,38)]
        n_experiments = 44
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                #path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.001-w235.61944901923448a0.012-'+str(k) 
                path = str(i)+'RhoMean'+tipo2+'lr0.001-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
    if tipo2 == "brain":
        #l = [i for i in range(1,33)]
        l = [i for i in range(1,30)]
        n_experiments = 43
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                #path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.001-w282.7433388230814a0.012-'+str(k)
                path = str(i)+'RhoMean'+tipo2+'lr0.001-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]
    if tipo2 == "3d":
        l = [i for i in range(1,30)]
        n_experiments = 42
        mean_values = []
        median_values = []
        min_values = []
        for i in l:
            temp = []
            for k in range(n_experiments):
                #path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) 
                path = str(i)+'RhoMean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                temp.append(array[-1])
                # Store results
            temp = np.array(temp)
            mean_values.append((i, np.mean(temp)))
            median_values.append((i, np.median(temp)))
            min_values.append((i, np.min(temp)))
        best_mean_function = min(mean_values, key=lambda x: x[1])
        # Find the function with the smallest median
        best_median_function = min(median_values, key=lambda x: x[1])
        best_min_function = min(min_values, key=lambda x: x[1])
        print ("rhomean -"+tipo2)
        print("mean", tipo2, "Time = ", best_mean_function[0], best_mean_function[1])
        print("median", tipo2, "Time = ", best_median_function[0], best_median_function[1])
        print("minimum", tipo2, "Time = ",best_min_function[0], best_min_function[1])
        return best_mean_function[0], best_median_function[0], best_min_function[0]


def plot2(tipo):
    tipo2 = "human"
    mean, median, minimum = bestTime(tipo2)
    meanGau, medianGau, minimumGau = bestTimeGau(tipo2)
    if tipo == "mean" and tipo2 == "human":
        l = [mean]
        lGau = [meanGau]
    if tipo == "median" and tipo2 == "human":
        l = [median]
        lGau = [medianGau]
    if tipo == "min" and tipo2 == "human":
        l = [minimum]
        lGau = [minimumGau]
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []
    for i, G in zip(l,lGau):
        for k in range(n_experiments):
            path = str(i)+'RhoMean'+tipo2+'lr0.001-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            values.append(array)
            path = 'Uni'+tipo2+'lr0.01-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesuni.append(array)
            path = 'FBGau'+tipo2+'lr0.01-w'+str(G)+'a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesnorm.append(array)
            
            
            # Store results
    applied_results = np.array(values)  
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)

    fig, axs = plt.subplots(1, 3, figsize =(13, 4))

    # Fill the region between min and max values
    axs[1].fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    axs[1].fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    axs[1].fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1].plot(median_values, color='blue', linewidth=2, label='RhoMean')
    axs[1].plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    axs[1].plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1].set_xlabel('iterations')
    #axs[1].set_ylabel('Loss')
    axs[1].set_yscale("log")
    axs[1].set_title(tipo2)
    axs[1].set_yticks([1e0,1e-4])
    axs[1].set_xticks([0,1000])
    axs[1].xaxis.set_label_coords(0.5, -0.05)
    axs[1].set_xlim([0,1000])
    #axs[1].legend()
    
    ##################################################################################################

    tipo2 = "brain"
    mean, median, minimum = bestTime(tipo2)
    meanGau, medianGau, minimumGau = bestTimeGau(tipo2)
    if tipo == "mean" and tipo2 == "brain":
        l = [mean]
        lGau = [meanGau]
    if tipo == "median" and tipo2 == "brain":
        l = [median]
        lGau = [medianGau]
    if tipo == "min" and tipo2 == "brain":
        l = [minimum]
        lGau = [minimumGau]
    #l = [12]
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []
    for i, G in zip(l,lGau):
        for k in range(n_experiments):
            path = str(i)+'RhoMean'+tipo2+'lr0.001-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            values.append(array)
            path = 'Uni'+tipo2+'lr0.01-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesuni.append(array)
            path = 'FBGau'+tipo2+'lr0.01-w'+str(G)+'a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesnorm.append(array)
            
            # Store results
    applied_results = np.array(values)  
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)



    # Fill the region between min and max values
    axs[0].fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    axs[0].fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    axs[0].fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[0].plot(median_values, color='blue', linewidth=2, label='RhoMean')
    axs[0].plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    axs[0].plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('Error')
    axs[0].set_yscale("log")
    axs[0].set_title(tipo2)
    axs[0].set_yticks([1e0,1e-4])
    axs[0].set_xticks([0,1000])
    axs[0].xaxis.set_label_coords(0.5, -0.05)
    axs[0].yaxis.set_label_coords(-0.05, 0.5)
    axs[0].set_xlim([0,1000])
    #axs[0].legend()

    ##################################################################################################


    tipo2 = "3d"
    mean, median, minimum, = bestTime(tipo2)
    meanGau, medianGau, minimumGau = bestTimeGau(tipo2)
    if tipo == "mean" and tipo2 == "3d":
        l = [mean]
        lGau = [meanGau]
    if tipo == "median" and tipo2 == "3d":
        l = [median]
        lGau = [medianGau]
    if tipo == "min" and tipo2 == "3d":
        mean, median, minimum, = bestTimeFix(tipo2)
        l = [minimum]
        lGau = [minimumGau]
    #l = [27]
    n_experiments = 37
    values = []
    valuesuni = []
    valuesnorm = []
    for i, G in zip(l,lGau):
        for k in range(n_experiments):
            path = 'Fix'+str(i)+'Mean'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            values.append(array)
            path = 'Uni'+tipo2+'lr0.01-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesuni.append(array)
            path = 'FBGau'+tipo2+'lr0.01-w'+str(G)+'a0.012-'+str(k) #0.009000001-'+str(k)
            with open('speedres/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            valuesnorm.append(array)
            
            # Store results
    applied_results = np.array(values)  
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)


    # Fill the region between min and max values
    axs[2].fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    axs[2].fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    axs[2].fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[2].plot(median_values, color='blue', linewidth=2, label='Design')
    axs[2].plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    axs[2].plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[2].set_xlabel('iterations')
    #axs[2].set_ylabel('Loss')
    axs[2].set_yscale("log")
    axs[2].set_title(tipo2)
    axs[2].set_yticks([1e0,1e-3])
    axs[2].set_xticks([0,1000])
    axs[2].legend(frameon=False, loc='upper right', fontsize='small')
    axs[2].xaxis.set_label_coords(0.5, -0.05)
    axs[2].set_xlim([0,1000])
    plt.tight_layout()
    plt.savefig('speedres/N'+tipo+'RhoMeanfinal.pdf')
    plt.show()

    
 
    return None



if __name__ == '__main__':
    #plotFix("mean")
    #plotFix("median")
    #plotFix("min")

    #plot2("mean")
    plot2("median")
    #plot2("min")

    
    
    

    
    
    
    

    
