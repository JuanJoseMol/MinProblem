import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib
import pickle

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




def plot(tipo2):

    with open('speedres/2Design'+tipo2+'.pickle', 'rb') as file:
      data1 = pickle.load(file)

    listT = []
    por = []
    for i in data1.values():
        listT.append(i[0])
        por.append(i[2])

    
    plt.plot(listT,por, color='blue')


    # Add labels, legend, and title
    plt.xlabel('T')
    plt.ylabel('%')
    #plt.yscale("log")
    plt.title(tipo2)
    #plt.yticks([1e0, 1e-4])
    #plt.xticks([0, 30])
    #plt.gca().xaxis.set_label_coords(0.5, -0.05)
    #plt.gca().yaxis.set_label_coords(-0.05, 0.5)

    # Show legend
    #plt.legend()


    ##################################################################################################


    plt.show()

    
 
    return None


def plot2(tipo2):


    if tipo2 == "brain":
        with open('hiper/Anor-brainsigma7.65460292359158.txt', 'r') as f:
            a = np.loadtxt(f)
        with open('hiper/Adesign-brain-T4.txt', 'r') as f:
            b = np.loadtxt(f)
    if tipo2 == "human":
        with open('hiper/Anor-humansigma6.828606839967962.txt', 'r') as f:
            a = np.loadtxt(f)
        with open('hiper/Adesign-human-T4.txt', 'r') as f:
            b = np.loadtxt(f)
    if tipo2 == "3d":
        with open('hiper/Anor-3dsigma2.62800852366956.txt', 'r') as f:
            a = np.loadtxt(f)
        with open('hiper/Adesign-3d-T4.txt', 'r') as f:
            b = np.loadtxt(f)

    
    plt.plot(a, label="normal")
    plt.plot(b, label="design")


    # Add labels, legend, and title
    plt.xlabel('iteration')
    plt.ylabel('$\sigma_a$')
    plt.yscale("log")
    plt.title(tipo2)
    #plt.yticks([1e0, 1e-4])
    #plt.xticks([0, 30])
    #plt.gca().xaxis.set_label_coords(0.5, -0.05)
    #plt.gca().yaxis.set_label_coords(-0.05, 0.5)

    # Show legend
    plt.legend()


    ##################################################################################################


    plt.show()


 
    return None


def plot3():


    fig, axs = plt.subplots(1, 3, figsize =(16, 6))
    

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)

            with open('speedres/2norBrain.pickle', 'rb') as file:
                data3 = pickle.load(file)
            data4 = {**data1, **data3}
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/2norHuman.pickle', 'rb') as file:
                data3 = pickle.load(file)
            data4 = {**data1, **data3}
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/2nor3d.pickle', 'rb') as file:
                data3 = pickle.load(file)
            data4 = {**data1, **data3}
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)
        print (len(data4.values()))
        print (len(data1.values()))
        for T, sigma in data4.values():

            path = 'nor-'+tipo2+'T'+str(T)+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            axs[k].plot(array, label="T"+str(T)+"-$\sigma$"+str(sigma.round(2)))
        axs[k].set_xlabel('iterations')
        
        axs[k].set_yscale("log")
        axs[k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[k].set_xticks([0,1000])
        axs[k].legend(fontsize=8, loc='upper right')  

    axs[0].set_ylabel('Error')
       
    plt.show()


    fig2, axs2 = plt.subplots(1, 3, figsize =(16, 6))
    

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'design-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            axs2[k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs2[k].set_xlabel('iterations')
        
        axs2[k].set_yscale("log")
        axs2[k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs2[k].set_xticks([0,1000])
        axs2[k].legend(fontsize=8, loc='upper right')  


    axs2[0].set_ylabel('Error')
     
    plt.show()

def plot4():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T, sigma in data1.values():
            if T == 10000:
                continue

            path = 'norm2-'+tipo2+'T'+str(T)+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="T"+str(T)+"-$\sigma$"+str(sigma.round(2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')
    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')


    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'design2-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()


def plot5():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(2,9)]
        for sigma in listw:

            path = 'normal2-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    
    axs[0,0].set_ylabel('Error')


    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'design2-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]

    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()


def plot6():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'normal4-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')



    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'design4-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()



def plot7():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/ej1Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/ej1Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/ej1Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'ej1-15000normal2-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')




    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/ej1Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/ej1Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/ej1Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'ej1-15000design2-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()


def plot8():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'ej1-30000normal2-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')





    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/3Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'ej1-30000design2-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()


def plot9():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'ej2-30000normal2-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')



    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        for T1, root, sigma1 in data2.values():
            path = 'ej2-30000design2-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()


def plot5fix():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'normal2-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    
    axs[0,0].set_ylabel('Error')


    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej2Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej2Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej2Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        print (data2.keys())
        data = [data2[2], data2[3], data2[4], data2[5], data2[6], data2[7]]
        for T1, root, sigma1 in data:
            path = 'fej2-15000design-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]

    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()


def plot6fix():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/4Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            listw = [(k+1)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'normal4-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')



    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej1Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej1Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej1Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        data = [data2[2], data2[3], data2[4], data2[5], data2[6], data2[7]]
        for T1, root, sigma1 in data:
            path = 'fej1-15000design-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()



def plotMedian(t1,t2,t3):
    tipo2 = "human"
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-human-T'+str(t2)+'lr0.05-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(10*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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

    
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-brain-T'+str(t1)+'lr0.05-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(20*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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

    n_experiments = 42
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-3d-T'+str(t3)+'lr0.005-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.005-w'+str(2*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[2].set_xlabel('iterations')#, labelpad=-55)
    #axs[2].set_ylabel('Loss')
    axs[2].set_yscale("log")
    axs[2].set_title(tipo2)
    axs[2].set_yticks([1e0,1e-3])
    axs[2].set_xticks([0,1000])
    axs[2].legend(frameon=False, loc='upper right', fontsize='small')
    axs[2].xaxis.set_label_coords(0.5, -0.05)
    axs[2].set_xlim([0,1000])
    plt.savefig('speedres/final/RegionB15-H05-3d025.pdf')
    plt.show()

    return None


def plot10():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    
    m = [[],[],[]]

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        

        if tipo2 == "3d":
            listw = [(k)*np.pi for k in range(1,9)]
        else:
            listw = [5*(k+1)*np.pi for k in range(1,9)]
        for sigma in listw:

            path = 'MLej1-15000normal-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')





    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":
            with open('speedres/3norBrain.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej1Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":
            with open('speedres/3norHuman.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej1Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":
            with open('speedres/3nor3d.pickle', 'rb') as file:
                data1 = pickle.load(file)
            with open('speedres/fej1Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        data = [data2[2], data2[3], data2[4], data2[5], data2[6], data2[7]]
        for T1, root, sigma1 in data:
            path = 'MLej1-15000design2-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
        axs[1,k].set_xlabel('iterations')
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--', label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--', label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--', label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[1,0].set_ylabel('Error')
     
    plt.show()

def plottest(n):
    if n ==1:
        lr = 0.05
    if n ==2:
        lr = 0.04
        s = "2"
    if n ==3:
        lr = 0.03
        s = "3"
    if n ==4:
        lr = 0.02
        s = "4"
    
    for k in range(5):
        if n ==1:
            path = "testej1-15000normal2-brainsigma31.41592653589793-key"+str(k)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
        else:
            path = "test"+s+"ej1-15000normal-brainsigma31.41592653589793-key"+str(k)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
        plt.plot(array)
    plt.yscale('log')
    plt.title("lr "+str(lr))
    plt.show()


def plotFinalEj1():


    fig, axs = plt.subplots(2, 3, figsize =(16, 10))
    colors = plt.cm.inferno(np.linspace(0,.8,4))
    
    m = [[],[],[]]
    v = np.pi

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [2*v,3*v,5*v,7*v]
        elif tipo2 == "brain":
            listw = [15*v, 25*v, 35*v, 45*v]
        elif tipo2 == "human":
            listw = [10*v, 20*v, 30*v, 40*v]
        for j, sigma in enumerate(listw):
            if tipo2 == "3d":
                path = 'f2-ej1-15000normal-'+tipo2+'sigma'+str(sigma)
            else:
                path = 'f-ej1-15000normal-'+tipo2+'sigma'+str(sigma)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, color = colors[3-j], label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        """
        if k==0 or k==1:
            
            axs[0, k].set_ylim([1e-4,1e0])
            axs[0, k].set_yticks([1e-4,1e0])
        else:
            axs[0, k].set_ylim([1e-3,1e0])
            axs[0, k].set_yticks([1e-3,1e0])
        """
            
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')



    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "brain":

            with open('speedres/fej1Designbrain.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "human":

            with open('speedres/fej1Designhuman.pickle', 'rb') as file:
                data2 = pickle.load(file)
        if tipo2 == "3d":

            with open('speedres/fej1Design3d.pickle', 'rb') as file:
                data2 = pickle.load(file)

        if tipo2 == "3d":
            data = [ data2[.25],data2[.5],data2[1], data2[3]]
        elif tipo2 == "brain":
            data = [data2[2], data2[4], data2[6], data2[8]]
        elif tipo2 == "human":
            data = [data2[1], data2[3], data2[5], data2[7]]
        j = 0
        for T1, root, sigma1 in data:
            if tipo2 == "3d":
                path = 'f2-ej1-15000design-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            else:
                path = 'f-ej1-15000design-'+tipo2+'T'+str(T1)+'por'+str(sigma1)
            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            m[k].append(np.min(array))
            axs[1,k].plot(array, color = colors[3-j], label="T"+str(T1)+"-$\%$"+str(np.round(sigma1,2)))
            j+=1
        axs[1,k].set_xlabel('iterations', labelpad=-15)
        """
        if k==0 or k==1:
            
            axs[1, k].set_ylim([1e-4,1e0])
            axs[1, k].set_yticks([1e-4,1e0])
        else:
            
            axs[1, k].set_ylim([1e-3,1e0])
            axs[1, k].set_yticks([1e-3,1e0])
        """
        
        axs[1,k].set_yscale("log")
        #axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
          

    real_m =[min(m[0]),min(m[1]),min(m[2])]
    axs[0,0].plot([real_m [0] for i in range(1000)], '--')#, label=str(real_m [0]))
    axs[0,1].plot([real_m [1] for i in range(1000)], '--')#, label=str(real_m [1]))
    axs[0,2].plot([real_m [2] for i in range(1000)], '--')#, label=str(real_m [2]))
    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')
    axs[1,0].plot([real_m [0] for i in range(1000)], '--')#, label=str(real_m [0]))
    axs[1,1].plot([real_m [1] for i in range(1000)], '--')#, label=str(real_m [1]))
    axs[1,2].plot([real_m [2] for i in range(1000)], '--')#, label=str(real_m [2]))
    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')
    axs[0, 0].set_xlim([0,1000])
    axs[0, 1].set_xlim([0,1000])
    axs[0, 2].set_xlim([0,1000])
    axs[1, 0].set_xlim([0,1000])
    axs[1, 1].set_xlim([0,1000])
    axs[1, 2].set_xlim([0,1000])
    
    

    axs[1,0].set_ylabel('Error')
    plt.tight_layout()
    plt.savefig('hiper/pej1NorVRdesign.pdf', bbox_inches='tight')
     
    plt.show()




def plotBestSigma():
    """
    learnigspeed("3d", [1*v,2*v,8.806968110818094,4*v,5*v] ,1000, 5*1e-3, m, ej)
    learnigspeed("3d", [0.25*v,0.5*v,1.972297069765665,1*v,2*v] ,1000, 5*1e-4, m, ej)
    learnigspeed("3d", [0.0625*v,0.125*v,0.7024330507416472,0.5*v,1*v] ,1000, 5*1e-5, m, ej)
    
    learnigspeed("human", [5*v,10*v,47.62657699221887,20*v,25*v] ,1000, 5*1e-2, m, ej)
    learnigspeed("human", [0.5*v,1*v,7.489913599384419,3*v,5*v] ,10000, 5*1e-3, m, ej)
    learnigspeed("human", [0.125*v,0.25*v,0.9874030101510249,1*v,2*v] ,10000, 5*1e-4, m, ej)
    
    learnigspeed("brain", [15*v,20*v,75.69107062944478,30*v,35*v] ,1000, 5*1e-2, m, ej)
    learnigspeed("brain", [1*v,3*v,15.5462334090731,7*v,9*v] ,10000, 5*1e-3, m, ej)
    learnigspeed("brain", [0.25*v,0.5*v,2.2015495125481706,1*v,2*v] ,10000, 5*1e-4, m, ej)
    """


    fig, axs = plt.subplots(3, 3, figsize =(12, 11))
    colors = plt.cm.plasma(np.linspace(0,.8,5))
    
    m = [[],[],[]]
    v = np.pi

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [0.25*v,0.5*v,1.972297069765665]
        elif tipo2 == "brain":
            listw = [15*v,20*v,75.69107062944478,30*v,35*v]
        elif tipo2 == "human":
            listw = [10*v,47.62657699221887,20*v,25*v]
        for j, sigma in enumerate(listw):
            path = 'sigma-ej1-15000normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[0,k].plot(array, color = colors[j], label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        """
        if k==0 or k==1:
            
            axs[0, k].set_ylim([1e-4,1e0])
            axs[0, k].set_yticks([1e-4,1e0])
        else:
            axs[0, k].set_ylim([1e-3,1e0])
            axs[0, k].set_yticks([1e-3,1e0])
        """
            
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[0,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[0,0].set_ylabel('Error')



    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [2*v,8.806968110818094,4*v,5*v]
        elif tipo2 == "brain":
            listw = [3*v,15.5462334090731,7*v,9*v]
        elif tipo2 == "human":
            listw = [0.5*v,7.489913599384419,3*v,5*v]
        for j, sigma in enumerate(listw):
            path = 'sigma-ej1-15000normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)
            m[k].append(np.min(array))

            axs[1,k].plot(array, color = colors[j], label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        """
        if k==0 or k==1:
            
            axs[0, k].set_ylim([1e-4,1e0])
            axs[0, k].set_yticks([1e-4,1e0])
        else:
            axs[0, k].set_ylim([1e-3,1e0])
            axs[0, k].set_yticks([1e-3,1e0])
        """
            
        axs[1,k].set_yscale("log")
        axs[1,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[1,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[1,0].set_ylabel('Error')

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [0.0625*v,0.125*v,0.7024330507416472,0.5*v,1*v]
        elif tipo2 == "brain":
            listw = [0.25*v,0.5*v,2.2015495125481706,1*v,2*v]
        elif tipo2 == "human":
            listw = [0.125*v,0.25*v,0.9874030101510249,1*v,2*v]
        for j, sigma in enumerate(listw):
            path = 'sigma-ej1-15000normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            axs[2,k].plot(array, color = colors[j], label="$\sigma$"+str(np.round(sigma,2)))
        #axs[0,k].set_xlabel('iterations')
        """
        if k==0 or k==1:
            
            axs[0, k].set_ylim([1e-4,1e0])
            axs[0, k].set_yticks([1e-4,1e0])
        else:
            axs[0, k].set_ylim([1e-3,1e0])
            axs[0, k].set_yticks([1e-3,1e0])
        """
            
        axs[2,k].set_yscale("log")
        axs[2,k].set_title(tipo2)
        #axs[k].set_yticks([1e0,1e-4])
        axs[2,k].set_xticks([0,1000])
        #axs[0,k].legend(fontsize=8, loc='upper right')  

    axs[2,0].set_ylabel('Error')
          


    axs[0,0].legend(fontsize=8, loc='upper right')
    axs[0,1].legend(fontsize=8, loc='upper right')
    axs[0,2].legend(fontsize=8, loc='upper right')

    axs[1,0].legend(fontsize=8, loc='upper right')
    axs[1,1].legend(fontsize=8, loc='upper right')
    axs[1,2].legend(fontsize=8, loc='upper right')

    axs[2,0].legend(fontsize=8, loc='upper right')
    axs[2,1].legend(fontsize=8, loc='upper right')
    axs[2,2].legend(fontsize=8, loc='upper right')
    axs[0, 0].set_xlim([0,1000])
    axs[0, 1].set_xlim([0,1000])
    axs[0, 2].set_xlim([0,1000])
    axs[1, 0].set_xlim([0,1000])
    axs[1, 1].set_xlim([0,1000])
    axs[1, 2].set_xlim([0,1000])
    axs[2, 0].set_xlim([0,1000])
    axs[2, 1].set_xlim([0,1000])
    axs[2, 2].set_xlim([0,1000])
    
    


    plt.tight_layout()
    plt.savefig('hiper/2.pdf', bbox_inches='tight')
     
    plt.show()

def plotBestSigma2():
    """
    sigma1
    [1*v,2*v,8.533289652054167,4*v,5*v] 3d
    [7*v,10*v,38.21,15*v,20*v] human
    [10*v,15*v,47.24,20*v,25*v] brain

    sigma2
    [1*v,2*v,9.073452123358896,4*v,5*v]
    [5*v,10*v,40.90306157837341,15*v,20*v]
    [12*v,16*v,57.57280992622677,25*v,30*v]
    """


    fig, axs = plt.subplots(1, 3, figsize =(12, 4))
    colors = plt.cm.plasma(np.linspace(0,.8,5))
    
    v = np.pi

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [2*v,2.5*v,9.073452123358896,4*v,5*v] # 1*v 1.5*v
        elif tipo2 == "brain":
            listw = [14*v,16*v,57.57280992622677,25*v,30*v] # 12*v
        elif tipo2 == "human":
            listw = [8*v,12*v,40.90306157837341,15*v,20*v] # 5*v,10*v
        for j, sigma in enumerate(listw):
            path = 'sigma2-ej1-normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)


            axs[k].plot(array, color = colors[j], label="$\sigma$"+str(np.round(sigma,2)))

            
        axs[k].set_yscale("log")
        axs[k].set_title(tipo2)

        axs[k].set_xticks([0,1000])


    axs[0].set_ylabel('Error')



    axs[0].legend(fontsize=8, loc='upper right')
    axs[1].legend(fontsize=8, loc='upper right')
    axs[2].legend(fontsize=8, loc='upper right')


    axs[0].set_xlim([0,1000])
    axs[1].set_xlim([0,1000])
    axs[2].set_xlim([0,1000])

    
    


    plt.tight_layout()
    #plt.savefig('hiper/2.pdf', bbox_inches='tight')
     
    plt.show()

def plotMedianCut(t1,t2,t3, cut1, cut2, cut3):
    tipo2 = "human"
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-human-T'+str(t2)+'lr0.05-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(15*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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

    fig, axs = plt.subplots(1, 3, figsize =(14, 5))

    # Fill the region between min and max values
    axs[1].fill_between(np.arange(len(min_values))[:cut2], min_values[:cut2], max_values[:cut2], color='lightblue', alpha=0.5)
    axs[1].fill_between(np.arange(len(min_values))[:cut2], min_valuesuni[:cut2], max_valuesuni[:cut2], color='pink', alpha=0.5)
    axs[1].fill_between(np.arange(len(min_values))[:cut2], min_valuesnorm[:cut2], max_valuesnorm[:cut2], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1].plot(median_values[:cut2], color='blue', linewidth=2, label='RhoMean')
    axs[1].plot(median_valuesuni[:cut2], color='red', linewidth=2, label='Uniform')
    axs[1].plot(median_valuesnorm[:cut2], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1].set_xlabel('iterations')
    #axs[1].set_ylabel('Loss')
    axs[1].set_yscale("log")
    axs[1].set_title(tipo2)
    axs[1].set_yticks([1e0,1e-3])
    axs[1].set_xticks([0,cut2])
    axs[1].xaxis.set_label_coords(0.5, -0.05)
    axs[1].set_xlim([0,cut2])
    #axs[1].legend()
    
    ##################################################################################################

    tipo2 = "brain"

    
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-brain-T'+str(t1)+'lr0.05-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(20*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[0].fill_between(np.arange(len(min_values))[:cut1], min_values[:cut1], max_values[:cut1], color='lightblue', alpha=0.5)
    axs[0].fill_between(np.arange(len(min_values))[:cut1], min_valuesuni[:cut1], max_valuesuni[:cut1], color='pink', alpha=0.5)
    axs[0].fill_between(np.arange(len(min_values))[:cut1], min_valuesnorm[:cut1], max_valuesnorm[:cut1], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[0].plot(median_values[:cut1], color='blue', linewidth=2, label='RhoMean')
    axs[0].plot(median_valuesuni[:cut1], color='red', linewidth=2, label='Uniform')
    axs[0].plot(median_valuesnorm[:cut1], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('Error')
    axs[0].set_yscale("log")
    axs[0].set_title(tipo2)
    axs[0].set_yticks([1e0,1e-4])
    axs[0].set_xticks([0,cut1])
    axs[0].xaxis.set_label_coords(0.5, -0.05)
    axs[0].yaxis.set_label_coords(-0.05, 0.5)
    axs[0].set_xlim([0,cut1])

    #axs[0].legend()

    ##################################################################################################


    tipo2 = "3d"

    n_experiments = 42
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-3d-T'+str(t3)+'lr0.005-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.005-w'+str(3*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[2].fill_between(np.arange(len(min_values))[:cut3], min_values[:cut3], max_values[:cut3], color='lightblue', alpha=0.5)
    axs[2].fill_between(np.arange(len(min_values))[:cut3], min_valuesuni[:cut3], max_valuesuni[:cut3], color='pink', alpha=0.5)
    axs[2].fill_between(np.arange(len(min_values))[:cut3], min_valuesnorm[:cut3], max_valuesnorm[:cut3], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[2].plot(median_values[:cut3], color='blue', linewidth=2, label='Design')
    axs[2].plot(median_valuesuni[:cut3], color='red', linewidth=2, label='Uniform')
    axs[2].plot(median_valuesnorm[:cut3], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[2].set_xlabel('iterations')#, labelpad=-55)
    #axs[2].set_ylabel('Loss')
    axs[2].set_yscale("log")
    axs[2].set_title(tipo2)
    axs[2].set_yticks([1e0,1e-3])
    axs[2].set_xticks([0,cut3])
    axs[2].legend(frameon=False, loc='upper right', fontsize=14)
    axs[2].xaxis.set_label_coords(0.5, -0.05)
    axs[2].set_xlim([0,cut3])
    plt.tight_layout()
    plt.savefig('speedres/final/cutRegionB15-H05-3d025.pdf')
    plt.show()

    return None

def plotMedianCut1(t1,t2,t3, cut1, cut2, cut3):

    """

    learnigspeed("3d", [1*v,2*v,8.510997196304478,4*v,5*v] ,1000, 5*1e-3, m, ej) #,2*v,9.073452123358896,4*v,5*v
    learnigspeed("human", [3*v,7*v,35.86792075220431,15*v,19*v] ,1000, 5*1e-2, m, ej) #,40.90306157837341,15*v,20*v
    learnigspeed("brain", [3*v,8*v,43.065079948177704,18*v,23*v] ,1000, 5*1e-2, m, ej) #,16*v,57.57280992622677,25*v,30*v
    """

    fig, axs = plt.subplots(2, 3, figsize =(13, 8)) #13, 8

    colors = plt.cm.plasma(np.linspace(0,.8,5))
    
    v = np.pi

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw =  [1.5*v,2*v,8.510997196304478,4*v,5*v] # 1*v 1.5*v
        elif tipo2 == "brain":
            listw = [11*v,13*v,43.065079948177704,20*v,30*v]# 12*v
        elif tipo2 == "human":
            listw = [6*v,8*v,35.86792075220431,15*v,19*v] # 5*v,10*v
        for j, sigma in enumerate(listw):
            path = 'sigma3-ej1-normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            if j == 2:
                axs[0,k].plot(array, color = colors[j], label="$\sigma_w="+str(np.round(sigma,2))+"$")
            else:
                axs[0,k].plot(array, '--', color = colors[j] ,label="$\sigma_w="+str(np.round(sigma,2))+"$")

            
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        
        axs[0,k].set_xticks([0,1000])
        axs[0,k].set_xlim([-10,1000])
        axs[0,k].set_xticks(axs[0,k].get_xlim())  # Use the current x-axis limits for ticks

        # Get tick positions and labels
        ticks = axs[0,k].get_xticks()
        print(ticks)
        ticks = [0, 1000]
        labels = axs[0,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        axs[0,k].legend(loc='upper right',frameon=False,fontsize=14)
    
    
    axs[0,0].set_yticks([1e-4,1e0])
    axs[0,1].set_yticks([1e-4,1e0])
    axs[0,2].set_yticks([1e-3,1e0])
    axs[0,0].set_ylim([1e-4,5*1e0])
    axs[0,1].set_ylim([1e-4,5*1e0])
    axs[0,2].set_ylim([1e-3,5*1e0])
    
    axs[0,0].yaxis.set_label_coords(-0.05, 0.5)
    

    axs[0,0].set_ylabel('Error')



    tipo2 = "human"
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-human-T'+str(t2)+'lr0.05-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(35.86792075220431)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_values[:cut2], max_values[:cut2], color='lightblue', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_valuesuni[:cut2], max_valuesuni[:cut2], color='pink', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_valuesnorm[:cut2], max_valuesnorm[:cut2], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,1].plot(median_values[:cut2], color='blue', linewidth=2, label='RhoMean')
    axs[1,1].plot(median_valuesuni[:cut2], color='red', linewidth=2, label='Uniform')
    axs[1,1].plot(median_valuesnorm[:cut2], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,1].set_xlabel('iterations')
    #axs[1].set_ylabel('Loss')
    axs[1,1].set_yscale("log")
    #axs[1,1].set_title(tipo2)
    axs[1,1].set_yticks([1e0,1e-3])
    axs[1,1].set_xticks([0,cut2])
    axs[1,1].xaxis.set_label_coords(0.5, -0.05)
    axs[1,1].set_xlim([-1,cut2])
    #axs[1].legend()
    
    ##################################################################################################

    tipo2 = "brain"

    
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-brain-T'+str(t1)+'lr0.05-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(15*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_values[:cut1], max_values[:cut1], color='lightblue', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesuni[:cut1], max_valuesuni[:cut1], color='pink', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesnorm[:cut1], max_valuesnorm[:cut1], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,0].plot(median_values[:cut1], color='blue', linewidth=2, label='RhoMean')
    axs[1,0].plot(median_valuesuni[:cut1], color='red', linewidth=2, label='Uniform')
    axs[1,0].plot(median_valuesnorm[:cut1], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,0].set_xlabel('iterations')
    axs[1,0].set_ylabel('Error')
    axs[1,0].set_yscale("log")
    #axs[1,0].set_title(tipo2)
    axs[1,0].set_yticks([1e0,1e-4])
    axs[1,0].set_xticks([0,cut1])
    axs[1,0].xaxis.set_label_coords(0.5, -0.05)
    axs[1,0].yaxis.set_label_coords(-0.05, 0.5)
    axs[1,0].set_xlim([-5,cut1])

    #axs[0].legend()

    ##################################################################################################


    tipo2 = "3d"

    n_experiments = 42
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'final/Design-3d-T'+str(t3)+'lr0.005-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.005-w'+str(8.510997196304478)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_values[:cut3], max_values[:cut3], color='lightblue', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_valuesuni[:cut3], max_valuesuni[:cut3], color='pink', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_valuesnorm[:cut3], max_valuesnorm[:cut3], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,2].plot(median_values[:cut3], color='blue', linewidth=2, label='Design')
    axs[1,2].plot(median_valuesuni[:cut3], color='red', linewidth=2, label='Uniform')
    axs[1,2].plot(median_valuesnorm[:cut3], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,2].set_xlabel('iterations')#, labelpad=-55)
    #axs[2].set_ylabel('Loss')
    axs[1,2].set_yscale("log")
    #axs[1,2].set_title(tipo2)
    axs[1,2].set_yticks([1e0,1e-3])
    axs[1,2].set_xticks([0,cut3])
    axs[1,2].legend(loc='upper right',frameon=False,fontsize=16)
    axs[1,2].xaxis.set_label_coords(0.5, -0.05)
    axs[1,2].set_xlim([-10,cut3])

    for k in range(3):
        axs[1,k].set_xticks(axs[1,k].get_xlim())
        if k==0:
            ticks = [0, cut1]
        if k==1:
            ticks = [0, cut2]
        if k==2:
            ticks = [0, cut3]
        labels = axs[1,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")

        #axs[1,k].legend(frameon=False, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.21, hspace=0.15)
    plt.savefig('speedres/final/figure_5.pdf')
    plt.show()

    return None

def plotMedianCut2(t1,t2,t3, cut1, cut2, cut3):

    fig, axs = plt.subplots(2, 3, figsize =(13, 8))

    colors = plt.cm.plasma(np.linspace(0,.8,5))
    
    v = np.pi
    """
    learnigspeed("3d", [1*v,2*v,8.55,4*v,5*v] ,1000, 5*1e-3, m, ej) #,2*v,9.073452123358896,4*v,5*v
    learnigspeed("human", [3*v,7*v,35.86,15*v,19*v] ,1000, 5*1e-2, m, ej) #,40.90306157837341,15*v,20*v
    learnigspeed("brain", [3*v,8*v,42.21,18*v,23*v] ,1000, 5*1e-2, m, ej) #,16*v,57.57280992622677,25*v,30*v
    """

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [1.5*v,2*v,8.55,4*v,5*v] # 1*v 1.5*v
        elif tipo2 == "brain":
            listw = [11*v,13*v,42.21,20*v,30*v] # 12*v
        elif tipo2 == "human":
            listw = [6*v,8*v,35.86,15*v,20*v]  # 5*v,10*v
        for j, sigma in enumerate(listw):
            path = 'sigma3-ej1-normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            if j == 2:
                axs[0,k].plot(array, color = colors[j], label="$\sigma_w="+str(np.round(sigma,2))+"$")
            else:
                axs[0,k].plot(array, '--', color = colors[j] ,label="$\sigma_w="+str(np.round(sigma,2))+"$")

        axs[0,k].set_xticks([0,1000])
        axs[0,k].set_xlim([-10,1000])
        
        axs[0,k].legend(loc='upper right',frameon=False,fontsize=14)
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        ticks = axs[0,k].get_xticks()
        labels = axs[0,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        #axs[0,k].legend(frameon=False, loc='upper right', fontsize=10)
        


    axs[0,0].set_yticks([1e0,1e-4])
    axs[0,1].set_yticks([1e0,1e-4])
    axs[0,2].set_yticks([1e0,1e-3])
    axs[0,0].set_ylim([1e-4,5*1e0])
    axs[0,1].set_ylim([1e-4,5*1e0])
    axs[0,2].set_ylim([1e-3,5*1e0])
    axs[0,0].set_ylabel('Error')
    axs[0,0].yaxis.set_label_coords(-0.05, 0.5)

    N=2
    tipo2 = "human"
    n_experiments = 44
    values = []
    valuesN = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'sgd'+str(t2)+'RhoMean'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = f"sgdN{N}RhoMean{tipo2}lr0.05-w235.61944901923448a0.012-{k}"
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesN.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(35.86)+'a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesnorm.append(array)
            
            
            # Store results
    applied_results = np.array(values)  
    applied_resultsN = np.array(valuesN) 
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesN = np.min(applied_resultsN, axis=0)  # Minimum values for each example
    max_valuesN = np.max(applied_resultsN, axis=0)  # Maximum values for each example
    median_valuesN = np.median(applied_resultsN, axis=0) 

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)

    

    # Fill the region between min and max values
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_values[:cut2], max_values[:cut2], color='lightblue', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_valuesN))[:cut2], min_valuesN[:cut2], max_valuesN[:cut2], color='lightblue', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_valuesuni[:cut2], max_valuesuni[:cut2], color='pink', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_valuesnorm[:cut2], max_valuesnorm[:cut2], color='lightgreen', alpha=0.5)

    
    axs[1,1].plot(median_values[:cut2], color='blue', linewidth=2, label='RhoMean')
    axs[1,1].plot(median_values[:cut2], color='yellow', linewidth=2, label='RhoMean')
    axs[1,1].plot(median_valuesuni[:cut2], color='red', linewidth=2, label='Uniform')
    axs[1,1].plot(median_valuesnorm[:cut2], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,1].set_xlabel('iterations')
    #axs[1].set_ylabel('Loss')
    axs[1,1].set_yscale("log")
    #axs[1,1].set_title(tipo2)
    axs[1,1].set_yticks([1e0,1e-3])
    axs[1,1].set_xticks([0,cut2])
    axs[1,1].xaxis.set_label_coords(0.5, -0.05)
    axs[1,1].set_xlim([-1,cut2])
    #axs[1].legend()
    
    ##################################################################################################

    tipo2 = "brain"

    N=2
    n_experiments = 44
    values = []
    valuesN = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'sgd'+str(t1)+'RhoMean'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = f"sgdN{N}RhoMean{tipo2}lr0.05-w282.7433388230814a0.012-{k}"
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesN.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(15*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesnorm.append(array)
            
            # Store results
    applied_results = np.array(values)  
    applied_resultsN = np.array(valuesN)
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesN = np.min(applied_resultsN, axis=0)  # Minimum values for each example
    max_valuesN = np.max(applied_resultsN, axis=0)  # Maximum values for each example
    median_valuesN = np.median(applied_resultsN, axis=0) 

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)



    # Fill the region between min and max values
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_values[:cut1], max_values[:cut1], color='lightblue', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesN[:cut1], max_valuesN[:cut1], color='lightblue', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesuni[:cut1], max_valuesuni[:cut1], color='pink', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesnorm[:cut1], max_valuesnorm[:cut1], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,0].plot(median_values[:cut1], color='blue', linewidth=2, label='RhoMean')
    axs[1,0].plot(median_valuesN[:cut1], color='yellow', linewidth=2, label='RhoMean')
    axs[1,0].plot(median_valuesuni[:cut1], color='red', linewidth=2, label='Uniform')
    axs[1,0].plot(median_valuesnorm[:cut1], color='green', linewidth=2, label='Normal')


    # Add labels, legend, and title
    axs[1,0].set_xlabel('iterations')
    axs[1,0].set_ylabel('Error')
    axs[1,0].set_yscale("log")
    #axs[1,0].set_title(tipo2)
    axs[1,0].set_yticks([1e0,1e-4])
    axs[1,0].set_xticks([0,cut1])
    axs[1,0].xaxis.set_label_coords(0.5, -0.05)
    axs[1,0].yaxis.set_label_coords(-0.05, 0.5)
    axs[1,0].set_xlim([-5,cut1])

    #axs[0].legend()

    ##################################################################################################


    tipo2 = "3d"
    N=1
    n_experiments = 42
    values = []
    valuesN = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'sgd'+str(t3)+'RhoMean'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = f"sgdN{N}RhoMean{tipo2}lr0.005-w70.68583470577035a0.012-{k}"
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesN.append(array)
        path = 'psgdUni'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.005-w'+str(8.55)+'a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesnorm.append(array)
            
            # Store results
    applied_results = np.array(values)  
    applied_resultsN = np.array(valuesN)
    uni = np.array(valuesuni)
    norm = np.array(valuesnorm)

    min_values = np.min(applied_results, axis=0)  # Minimum values for each example
    max_values = np.max(applied_results, axis=0)  # Maximum values for each example
    median_values = np.median(applied_results, axis=0)  

    min_valuesN = np.min(applied_resultsN, axis=0)  # Minimum values for each example
    max_valuesN = np.max(applied_resultsN, axis=0)  # Maximum values for each example
    median_valuesN = np.median(applied_resultsN, axis=0) 

    min_valuesuni = np.min(uni, axis=0)  # Minimum values for each example
    max_valuesuni = np.max(uni, axis=0)  # Maximum values for each example
    median_valuesuni = np.median(uni, axis=0)

    min_valuesnorm = np.min(norm, axis=0)  # Minimum values for each example
    max_valuesnorm = np.max(norm, axis=0)  # Maximum values for each example
    median_valuesnorm = np.median(norm, axis=0)


    # Fill the region between min and max values
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_values[:cut3], max_values[:cut3], color='lightblue', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut1], min_valuesN[:cut1], max_valuesN[:cut1], color='lightblue', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_valuesuni[:cut3], max_valuesuni[:cut3], color='pink', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_valuesnorm[:cut3], max_valuesnorm[:cut3], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,2].plot(median_values[:cut3], color='blue', linewidth=2, label='N=450 Design')
    axs[1,2].plot(median_valuesN[:cut3], color='yellow', linewidth=2, label='N=2 Design')
    axs[1,2].plot(median_valuesuni[:cut3], color='red', linewidth=2, label='Uniform')
    axs[1,2].plot(median_valuesnorm[:cut3], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,2].set_xlabel('iterations')#, labelpad=-55)
    #axs[2].set_ylabel('Loss')
    axs[1,2].set_yscale("log")
    #axs[1,2].set_title(tipo2)
    axs[1,2].set_yticks([1e0,1e-3])
    axs[1,2].set_xticks([0,cut3])
    axs[1,2].legend(loc='upper right',frameon=False,fontsize=13)
    axs[1,2].xaxis.set_label_coords(0.5, -0.05)
    axs[1,2].set_xlim([-10,cut3])

    for k in range(3):
        axs[1,k].set_xticks(axs[1,k].get_xlim())
        if k==0:
            ticks = [0, cut1]
        if k==1:
            ticks = [0, cut2]
        if k==2:
            ticks = [0, cut3]
        labels = axs[1,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        #axs[1,k].legend(frameon=False, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.21, hspace=0.15)
    plt.savefig('speedres/final/figure_6.pdf')
    plt.show()

    return None


def plotMedianCut3(t1,t2,t3, cut1, cut2, cut3):

    fig, axs = plt.subplots(3, 3, figsize =(12, 10))

    colors = plt.cm.plasma(np.linspace(0,.8,5))
    
    v = np.pi
    """
    learnigspeed("3d", [1*v,2*v,8.55,4*v,5*v] ,1000, 5*1e-3, m, ej) #,2*v,9.073452123358896,4*v,5*v
    learnigspeed("human", [3*v,7*v,35.86,15*v,19*v] ,1000, 5*1e-2, m, ej) #,40.90306157837341,15*v,20*v
    learnigspeed("brain", [3*v,8*v,42.21,18*v,23*v] ,1000, 5*1e-2, m, ej) #,16*v,57.57280992622677,25*v,30*v
    """

    for k, tipo2 in enumerate(["brain", "human", "3d"]):
        if tipo2 == "3d":
            listw = [1.5*v,2*v,8.55,4*v,5*v] # 1*v 1.5*v
        elif tipo2 == "brain":
            listw = [11*v,13*v,42.21,20*v,30*v] # 12*v
        elif tipo2 == "human":
            listw = [6*v,8*v,35.86,15*v,20*v]  # 5*v,10*v
        for j, sigma in enumerate(listw):
            path = 'sigma3-ej1-normal-'+tipo2+'sigma'+str(sigma)

            with open('hiper/'+path+'.txt', 'r') as f:
                array = np.loadtxt(f)

            if j == 2:
                if sigma == 8.55:
                    sigma = 8.84
                if sigma == 42.21:
                    sigma = 43.02
                if sigma == 35.86:
                    sigma = 36.65
                axs[0,k].plot(array, color = colors[j], label="$\sigma_w="+str(np.round(sigma,2))+"$")
            else:
                axs[0,k].plot(array, '--', color = colors[j] ,label="$\sigma_w="+str(np.round(sigma,2))+"$")

        axs[0,k].set_xticks([0,1000])
        axs[0,k].set_xlim([-10,1000])
        
        axs[0,k].legend(loc='upper right',frameon=False,fontsize=14)
        axs[0,k].set_yscale("log")
        axs[0,k].set_title(tipo2)
        ticks = axs[0,k].get_xticks()
        labels = axs[0,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        #axs[0,k].legend(frameon=False, loc='upper right', fontsize=10)
        


    axs[0,0].set_yticks([1e0,1e-4])
    axs[0,1].set_yticks([1e0,1e-4])
    axs[0,2].set_yticks([1e0,1e-3])
    axs[0,0].set_ylim([1e-4,5*1e0])
    axs[0,1].set_ylim([1e-4,5*1e0])
    axs[0,2].set_ylim([1e-3,5*1e0])
    axs[0,0].set_ylabel('Error')
    axs[0,0].yaxis.set_label_coords(-0.05, 0.5)

        


    tipo2 = "human"
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'sgd'+str(t2)+'RhoMean'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w235.61944901923448a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(35.86)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_values[:cut2], max_values[:cut2], color='lightblue', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_valuesuni[:cut2], max_valuesuni[:cut2], color='pink', alpha=0.5)
    axs[1,1].fill_between(np.arange(len(min_values))[:cut2], min_valuesnorm[:cut2], max_valuesnorm[:cut2], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,1].plot(median_values[:cut2], color='blue', linewidth=2, label='RhoMean')
    axs[1,1].plot(median_valuesuni[:cut2], color='red', linewidth=2, label='Uniform')
    axs[1,1].plot(median_valuesnorm[:cut2], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,1].set_xlabel('iterations')
    #axs[1].set_ylabel('Loss')
    axs[1,1].set_yscale("log")
    #axs[1,1].set_title(tipo2)
    axs[1,1].set_yticks([1e0,1e-3])
    axs[1,1].set_xticks([0,cut2])
    axs[1,1].xaxis.set_label_coords(0.5, -0.05)
    axs[1,1].set_xlim([-1,cut2])
    #axs[1].legend()
    
    ##################################################################################################

    tipo2 = "brain"

    
    n_experiments = 44
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'sgd'+str(t1)+'RhoMean'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.05-w282.7433388230814a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.05-w'+str(15*np.pi)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_values[:cut1], max_values[:cut1], color='lightblue', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesuni[:cut1], max_valuesuni[:cut1], color='pink', alpha=0.5)
    axs[1,0].fill_between(np.arange(len(min_values))[:cut1], min_valuesnorm[:cut1], max_valuesnorm[:cut1], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,0].plot(median_values[:cut1], color='blue', linewidth=2, label='RhoMean')
    axs[1,0].plot(median_valuesuni[:cut1], color='red', linewidth=2, label='Uniform')
    axs[1,0].plot(median_valuesnorm[:cut1], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,0].set_xlabel('iterations')
    axs[1,0].set_ylabel('Error')
    axs[1,0].set_yscale("log")
    #axs[1,0].set_title(tipo2)
    axs[1,0].set_yticks([1e0,1e-4])
    axs[1,0].set_xticks([0,cut1])
    axs[1,0].xaxis.set_label_coords(0.5, -0.05)
    axs[1,0].yaxis.set_label_coords(-0.05, 0.5)
    axs[1,0].set_xlim([-5,cut1])

    #axs[0].legend()

    ##################################################################################################


    tipo2 = "3d"

    n_experiments = 42
    values = []
    valuesuni = []
    valuesnorm = []

    for k in range(n_experiments):
        path = 'sgd'+str(t3)+'RhoMean'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        values.append(array)
        path = 'psgdUni'+tipo2+'lr0.005-w70.68583470577035a0.012-'+str(k) #0.009000001-'+str(k)
        with open('speedres/'+path+'.txt', 'r') as f:
            array = np.loadtxt(f)
        valuesuni.append(array)
        path = 'sgdGau'+tipo2+'lr0.005-w'+str(8.55)+'a0.012-'+str(k) #0.009000001-'+str(k)
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
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_values[:cut3], max_values[:cut3], color='lightblue', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_valuesuni[:cut3], max_valuesuni[:cut3], color='pink', alpha=0.5)
    axs[1,2].fill_between(np.arange(len(min_values))[:cut3], min_valuesnorm[:cut3], max_valuesnorm[:cut3], color='lightgreen', alpha=0.5)

    # Plot the min, max, and median lines
    #plt.plot(min_values, color='blue', linestyle='--', label='Minimum')
    #plt.plot(max_values, color='red', linestyle='--', label='Maximum')
    axs[1,2].plot(median_values[:cut3], color='blue', linewidth=2, label='MeanDesign')
    axs[1,2].plot(median_valuesuni[:cut3], color='red', linewidth=2, label='Uniform')
    axs[1,2].plot(median_valuesnorm[:cut3], color='green', linewidth=2, label='Normal')

    # Add labels, legend, and title
    axs[1,2].set_xlabel('iterations')#, labelpad=-55)
    #axs[2].set_ylabel('Loss')
    axs[1,2].set_yscale("log")
    #axs[1,2].set_title(tipo2)
    axs[1,2].set_yticks([1e0,1e-3])
    axs[1,2].set_xticks([0,cut3])
    axs[1,2].legend(loc='upper right',frameon=False,fontsize=16)
    axs[1,2].xaxis.set_label_coords(0.5, -0.05)
    axs[1,2].set_xlim([-10,cut3])

    for k in range(3):
        axs[1,k].set_xticks(axs[1,k].get_xlim())
        if k==0:
            ticks = [0, cut1]
        if k==1:
            ticks = [0, cut2]
        if k==2:
            ticks = [0, cut3]
        labels = axs[1,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        #axs[1,k].legend(frameon=False, loc='upper right', fontsize=10)

    tipo2 = "human"
    N_list = [4, 64]
    n_experiments = 44

    for N in N_list:
        values = []

        for k in range(n_experiments):
            path = f"sgdN{N}RhoMean{tipo2}lr0.05-w235.61944901923448a0.012-{k}"
            try:
                with open('speedres/' + path + '.txt', 'r') as f:
                    array = np.loadtxt(f)
                values.append(array)
            except:
                pass

        if len(values) == 0:
            continue

        applied_results = np.array(values)

        min_values = np.min(applied_results, axis=0)
        max_values = np.max(applied_results, axis=0)
        median_values = np.median(applied_results, axis=0)

        axs[2,1].fill_between(
            np.arange(len(min_values))[:cut2],
            min_values[:cut2],
            max_values[:cut2],
            alpha=0.15
        )
        axs[2,1].plot(
            median_values[:cut2],
            linewidth=2,
            label=f"N={N}"
        )

    axs[2,1].set_xlabel("iterations")
    
    axs[2,1].set_yscale("log")
    axs[2,1].set_xticks([0, cut2])
    axs[2,1].set_xlim([-5, cut2])
    axs[2,1].set_yticks([1e0,1e-3])
    axs[2,1].xaxis.set_label_coords(0.5, -0.05)
    axs[2,1].yaxis.set_label_coords(-0.05, 0.5)
    #axs[2,1].legend(frameon=False)
    

    tipo2 = "brain"
    N_list = [4, 64]
    n_experiments = 44

    for N in N_list:
        values = []

        for k in range(n_experiments):
            path = f"sgdN{N}RhoMean{tipo2}lr0.05-w282.7433388230814a0.012-{k}"
            try:
                with open('speedres/' + path + '.txt', 'r') as f:
                    array = np.loadtxt(f)
                values.append(array)
            except:
                pass

        if len(values) == 0:
            continue

        applied_results = np.array(values)

        min_values = np.min(applied_results, axis=0)
        max_values = np.max(applied_results, axis=0)
        median_values = np.median(applied_results, axis=0)

        axs[2,0].fill_between(
            np.arange(len(min_values))[:cut1],
            min_values[:cut1],
            max_values[:cut1],
            alpha=0.15
        )
        axs[2,0].plot(
            median_values[:cut1],
            linewidth=2,
            label=f"N={N}"
        )

    axs[2,0].set_xlabel("iterations")
    axs[2,0].set_ylabel("Error")
    axs[2,0].set_yscale("log")
    axs[2,0].set_xticks([0, cut1])
    axs[2,0].set_xlim([-5, cut1])
    axs[2,0].xaxis.set_label_coords(0.5, -0.05)
    axs[2,0].yaxis.set_label_coords(-0.05, 0.5)
    axs[2,0].set_yticks([1e0,1e-4])
    #axs[2,0].legend(frameon=False)

    tipo2 = "3d"
    N_list = [4, 64]
    n_experiments = 42

    for N in N_list:
        values = []

        for k in range(n_experiments):
            path = f"sgdN{N}RhoMean{tipo2}lr0.005-w70.68583470577035a0.012-{k}"
            try:
                with open('speedres/' + path + '.txt', 'r') as f:
                    array = np.loadtxt(f)
                values.append(array)
            except:
                pass

        if len(values) == 0:
            continue

        applied_results = np.array(values)

        min_values = np.min(applied_results, axis=0)
        max_values = np.max(applied_results, axis=0)
        median_values = np.median(applied_results, axis=0)

        axs[2,2].fill_between(
            np.arange(len(min_values))[:cut3],
            min_values[:cut3],
            max_values[:cut3],
            alpha=0.15
        )
        axs[2,2].plot(
            median_values[:cut3],
            linewidth=2,
            label=f"N={N}"
        )

    axs[2,2].set_xlabel("iterations")
    axs[2,2].set_yscale("log")
    axs[2,2].set_xticks([0, cut3])
    axs[2,2].set_xlim([-10, cut3])
    axs[2,2].xaxis.set_label_coords(0.5, -0.05)
    axs[2,2].yaxis.set_label_coords(-0.05, 0.5)
    axs[2,2].set_yticks([1e0,1e-3])
    axs[2,2].legend(frameon=False)

    for k in range(3):
        axs[2,k].set_xticks(axs[2,k].get_xlim())
        if k==0:
            ticks = [0, cut1]
        if k==1:
            ticks = [0, cut2]
        if k==2:
            ticks = [0, cut3]
        labels = axs[2,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        #axs[1,k].legend(frameon=False, loc='upper right', fontsize=10)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.21, hspace=0.15)
    plt.savefig('speedres/final/f3cutRegionB15-H05-3d025.pdf')
    plt.show()

    return None


if __name__ == '__main__':
    """
    tipo = "brain"
    plot(tipo)
    tipo = "human"
    plot(tipo)
    tipo = "3d"
    plot(tipo)

    tipo = "brain"
    plot2(tipo)
    tipo = "human"
    plot2(tipo)
    tipo = "3d"
    plot2(tipo)
    
    plot5()
    plot6()
    plot7()
    plot8()
    plot9()
    
    plot5fix()
    plot6fix()
    
    plot10()
    """
    #plotMedian(1.5,0.5,0.25)
    #plotMedianCut1(1.5,0.5,0.25, 400, 200, 1000)
    #plot5()
    #plottest(4)
    #plotFinalEj1()
    #plotBestSigma()
    #plotBestSigma2()
    plotMedianCut1(1.5,0.5,0.25, 400, 200, 1000)
    plotMedianCut2(1.8,0.75,1.2, 400, 200, 1000)

    
    

    
    
    
    

    
