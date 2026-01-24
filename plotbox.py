import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



# This function reads the files and returns a list of arrays

def read_experiment_data(tipo2, time, lr):
    if tipo2 == "human":
        l = ['10Mean', '20Mean', '10Medoid', '20Medoid', '10RhoMean', '20RhoMean', '10RhoMediod', '20RhoMediod']
        #l = ['MeanUni', '0.5MeanConst','1MeanConst', '5MeanConst','10MeanConst']
        n_experiments = 44
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)249.75661596038856
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array[time])
        return data
    if tipo2 == "brain":
        l = ['10Mean', '20Mean', '10Medoid', '20Medoid', '10RhoMean', '20RhoMean', '10RhoMediod', '20RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array[time])
        return data
    if tipo2 == "3d":
        l = ['10Mean', '20Mean', '10Medoid', '20Medoid', '10RhoMean', '20RhoMean', '10RhoMediod', '20RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w70.68583470577035a0.012-'+str(k) #0.010000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array[time])
        return data

def read_data_vector(tipo2,lr):
    if tipo2 == "human":
        l = ['10Mean', '20Mean', '30Mean', '10Medoid', '20Medoid', '30Medoid',
             '10RhoMean', '20RhoMean', '30RhoMean', '10RhoMediod', '20RhoMediod', '30RhoMediod']
        #l = ['MeanUni', '0.5MeanConst','1MeanConst', '5MeanConst','10MeanConst']
        n_experiments = 44
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean
    if tipo2 == "brain":
        l = ['10Mean', '20Mean', '25Mean', '10Medoid', '20Medoid', '25Medoid',
             '10RhoMean', '20RhoMean', '25RhoMean', '10RhoMediod', '20RhoMediod', '25RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean
    if tipo2 == "3d":
        l = ['10Mean', '20Mean', '30Mean', '10Medoid', '20Medoid', '30Medoid',
             '10RhoMean', '20RhoMean', '30RhoMean', '10RhoMediod', '20RhoMediod', '30RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w70.68583470577035a0.012-'+str(k) #0.010000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean


def read_data_vector_median(tipo2,clase,lr):
    if tipo2 == "human":
        if clase == "all":
            l = ['20Mean', '5Medoid', '5RhoMean','5RhoMediod']
        if clase == "Mean":
            l = ['5Mean','10Mean', '20Mean', '30Mean']
        if clase == "Mediod":
            l = ['5Medoid','10Medoid', '20Medoid', '30Medoid']
        if clase == "RhoMean":
            l = ['5RhoMean','10RhoMean', '20RhoMean', '30RhoMean']
        if clase == "RhoMediod":
            l = ['5RhoMediod','10RhoMediod', '20RhoMediod', '30RhoMediod']
        #l = ['10Mean', '20Mean', '30Mean', '10Medoid', '20Medoid', '30Medoid',
        #     '10RhoMean', '20RhoMean', '30RhoMean', '10RhoMediod', '20RhoMediod', '30RhoMediod']
        #l = ['MeanUni', '0.5MeanConst','1MeanConst', '5MeanConst','10MeanConst']
        n_experiments = 44
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean
    if tipo2 == "brain":
        if clase == "all":
            l = ['5Mean', '5Medoid', '5RhoMean','5RhoMediod']
        if clase == "Mean":
            l = ['5Mean','10Mean', '20Mean', '25Mean']
        if clase == "Mediod":
            l = ['5Medoid','10Medoid', '20Medoid', '25Medoid']
        if clase == "RhoMean":
            l = ['5RhoMean','10RhoMean', '20RhoMean', '25RhoMean']
        if clase == "RhoMediod":
            l = ['5RhoMediod','10RhoMediod', '20RhoMediod', '25RhoMediod']
        #l = ['10Mean', '20Mean', '25Mean', '10Medoid', '20Medoid', '25Medoid',
        #     '10RhoMean', '20RhoMean', '25RhoMean', '10RhoMediod', '20RhoMediod', '25RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean
    if tipo2 == "3d":
        if clase == "all":
            l = ['10Mean', '10Medoid', '10RhoMean','10RhoMediod']
        if clase == "Mean":
            l = ['5Mean','10Mean', '20Mean', '30Mean']
        if clase == "Mediod":
            l = ['5Medoid','10Medoid', '20Medoid', '30Medoid']
        if clase == "RhoMean":
            l = ['5RhoMean','10RhoMean', '20RhoMean', '30RhoMean']
        if clase == "RhoMediod":
            l = ['5RhoMediod','10RhoMediod', '20RhoMediod', '30RhoMediod']
        #l = ['10Mean', '20Mean', '30Mean', '10Medoid', '20Medoid', '30Medoid',
        #     '10RhoMean', '20RhoMean', '30RhoMean', '10RhoMediod', '20RhoMediod', '30RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w70.68583470577035a0.012-'+str(k) #0.010000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean

# This function generates a boxplot for one set of results
def plot_boxplot(tipo2, time, lr):

    data = read_experiment_data(tipo2, time, lr)

    fig = plt.figure(figsize =(12, 7))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    plt.title(tipo2+"- iteration "+str(time))
    # Creating axes instance
    #ax.set_xticklabels(["Uni", "ConstT1", "ConstT2", "ConstT3", "ConstT4"])
    ax.set_xticklabels(["MeanT1", "MeanT2", "MediodT1", "MediodT2", "RhoMeanT1", "RhoMeanT2", "RhoMediodT1", "RhoMediodT2"])
    

    # Creating plot
    bp = ax.boxplot(data)

    # show plot
    plt.show()

def plot_means(lr):
    colors = plt.cm.hsv(np.linspace(0,1,20))
    c = [colors[i] for i in range(0,20)]
    l = ["MeanT1", "MeanT2", "MeanT3", "MediodT1", "MediodT2", "MediodT3",
             "RhoMeanT1", "RhoMeanT2", "RhoMeanT3", "RhoMediodT1", "RhoMediodT2", "RhoMediodT3"]
    tip = ["brain", "human", "3d"]
    fig, axs = plt.subplots(1, 3, figsize =(14, 5))
    for j, k in enumerate(tip):
        data = read_data_vector(k,lr)
        for i in range(len(data)):
            axs[j].plot(np.linspace(0,1000,1000), data[i,:], color=c[i], label = l[i])
        axs[j].set_title(k)
        axs[j].set_yscale('log')
    # show plot
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('speedres/Imeans.pdf')
    plt.show()

def plot_median(clase, lr):
    colors = plt.cm.hsv(np.linspace(0,1,20))
    c = [colors[i] for i in range(0,20,5)]
    if clase == "all":
        l = ["Mean", "Mediod",  "RhoMean", "RhoMediod"]
    if clase == "Mean":
        l = ["MeanT1", "MeanT2", "MeanT3", "MeanT4"]
    if clase == "Mediod":
        l = ["MediodT1", "MediodT2", "MediodT3", "MediodT4"]
    if clase == "RhoMean":
        l = ["RhoMeanT1", "RhoMeanT2", "RhoMeanT3", "RhoMeanT4"]
    if clase == "RhoMediod":
        l = ["RhoMediodT1", "RhoMediodT2", "RhoMediodT3", "RhoMediodT4"]
    #l = ["MeanT1", "MeanT2", "MeanT3", "MeanT4", "MediodT1", "MediodT2", "MediodT3", "MediodT4",
    #         "RhoMeanT1", "RhoMeanT2", "RhoMeanT3", "RhoMeanT4", "RhoMediodT1", "RhoMediodT2", "RhoMediodT3", "RhoMediodT4"]
    tip = ["brain", "human", "3d"]
    fig, axs = plt.subplots(1, 3, figsize =(14, 5))
    for j, k in enumerate(tip):
        data = read_data_vector_median(k,clase,lr)
        for i in range(len(data)):
            axs[j].plot(np.linspace(0,1000,1000), data[i,:], color=c[i], label = l[i])
        axs[j].set_title(k)
        axs[j].set_yscale('log')
    # show plot
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('speedres/Imean'+clase+'.pdf')
    plt.show()

def plot_rhos(lr):
    colors = plt.cm.plasma(np.linspace(0,.9,21))
    c = [colors[0],colors[5],colors[10],colors[15]]
    l = ["RhoMeanT1", "RhoMeanT2", "RhoMediodT1", "RhoMediodT2"]
    tip = ["brain", "human", "3d"]
    fig, axs = plt.subplots(1, 3, figsize =(14, 5))
    for j, k in enumerate(tip):
        data = read_data_vector(k,lr)[5:]
        for i in range(len(data)):
            axs[j].plot(np.linspace(0,1000,1000), data[i,:], label = l[i])
        axs[j].set_title(k)
        axs[j].set_yscale('log')
    # show plot
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('speedres/Imeans2.pdf')
    plt.show()


def read_eps(tipo,tipo2,lr):
    if tipo2 == "human":
        l = ['10Mean', '20Mean', '10Medoid', '20Medoid', '10RhoMean', '20RhoMean', '10RhoMediod', '20RhoMediod']
        #l = ['MeanUni', '0.5MeanConst','1MeanConst', '5MeanConst','10MeanConst']
        n_experiments = 44
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean
    if tipo2 == "brain":
        l = ['10Mean', '20Mean', '10Medoid', '20Medoid', '10RhoMean', '20RhoMean', '10RhoMediod', '20RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean
    if tipo2 == "3d":
        l = ['10Mean', '20Mean', '10Medoid', '20Medoid', '10RhoMean', '20RhoMean', '10RhoMediod', '20RhoMediod']
        #l = ['MeanUni', '5MeanConst','10MeanConst', '15MeanConst','20MeanConst']
        n_experiments = 43
        data = [[] for i in range(len(l))]
        for i, tipo in enumerate(l):
            for k in range(n_experiments):
                path = tipo+tipo2+'lr'+str(lr)+'-w70.68583470577035a0.012-'+str(k) #0.010000001-'+str(k)
                with open('speedres/I'+path+'.txt', 'r') as f:
                    array = np.loadtxt(f)
                data[i].append(array)
        data = np.array(data)
        print(data.shape)
        mean = np.mean(data, axis=1)
        print (mean.shape)
        return mean

if __name__ == '__main__':
    tipo = "human"
    lr = 0.05
    #plot_boxplot("human", 999, 0.01)
    #plot_boxplot("3d", 999, 0.1)
    """
    plot_boxplot(tipo, 100, lr)
    plot_boxplot(tipo, 500, lr)
    plot_boxplot(tipo, 750, lr)
    plot_boxplot(tipo, 999, lr)
    """

    #plot_rhos()
    """
    plot_median("Mean",lr) #"Mean", "Mediod",  "RhoMean", "RhoMediod"
    plot_median("Mediod",lr)
    plot_median("RhoMean",lr)
    plot_median("RhoMediod",lr)
    """
    plot_median("all",lr)
