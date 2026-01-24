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




def plot(lr):

    tipo2 ="brain"
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'Longsgd'+str(0.5)+'RhoMean'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'LongsgdUni'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'LongsgdGau'+tipo2+'lr0.05-w'+str(62.83185307179586)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    plt.fill_between(np.arange(len(min_values)), min_values, max_values, color='lightblue', alpha=0.5)
    plt.fill_between(np.arange(len(min_values)), min_valuesuni, max_valuesuni, color='pink', alpha=0.5)
    plt.fill_between(np.arange(len(min_values)), min_valuesnorm, max_valuesnorm, color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    plt.plot(median_values, color='blue', linewidth=2, label='RhoMean')
    plt.plot(median_valuesuni, color='red', linewidth=2, label='Uniform')
    plt.plot(median_valuesnorm, color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale("log")
    plt.title(tipo2)
    plt.yticks([1e0, 1e-4])
    plt.xticks([0, 5000])
    plt.gca().xaxis.set_label_coords(0.5, -0.05)
    plt.gca().yaxis.set_label_coords(-0.05, 0.5)

    # Show legend
    plt.legend()


    ##################################################################################################


    plt.show()

    
 
    return None




if __name__ == '__main__':
    plot(0.001)
    
    
    

    
    
    
    

    
