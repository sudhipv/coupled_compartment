'''
Using a parallel TMCMC sampler from "https://github.com/mukeshramancha/transitional-mcmc/tree/main"
'''
#### Code to infer many parameters for a single PHU

#!/usr/bin/python
import os, math, sys, random
import numpy as np
import numpy.linalg as la
import scipy.stats as st
import scipy.optimize as sopt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# import tmcmc_alpha
# From opensource repo:
from tmcmc_mod import pdfs
from tmcmc_mod.tmcmc_mod import run_tmcmc

np.random.seed(106)  # fixing the random seed

### CHANGE HERE : IF YOU ARE RUNNING ON USER MACHINE/LAPTOP USE MULTIPROCESSING AND IF YOU ARE RUNNING ON CLUSTERS USE MPI.
parallel_processing = 'multiprocessing' #'multiprocessing','mpi'


#### CHANGE HERE ############

## Number of parameters to infer
Npar = 10 # number of unknown parameters


############# REAL DATA ######################

phiTrue = np.zeros([Npar])

### SETTING THE LOWER AND UPPER BOUNDS BASED ON MANUAL TUNING. JUST HAVING KNOWLEDGE IF THE PARAMETER IS POSITIVE OR NEGATIVE
x_MLE_low = np.zeros([10])
x_MLE_up = np.zeros([10])

# Synthetic truth
# a0 =   0.14
# a1 =  -0.115
# t1 =   20
# a2 =   0.06
# t2 =  35
# a3 =   -0.05
# t3 = 65
# a4 =  0.035
# t4 = 90
# a5 =  0.06
# t5 = 140
# a6 =  -0.075
# t6 = 180
# a7 =  0.06
# t7 = 190
# a8 =  -0.025
# t8 = 250

# 1.30772565e-01 -8.54004108e-02  2.65404473e-02 -3.35566339e-02
#   3.42296008e-02  5.76184965e-02 -9.91353950e-02  1.29417795e-01
#   9.59998348e+01  9.59998105e+01]


bval = 0.2


x_MLE_low = [0,                    0,                1.30772565e-01  - bval, -8.54004108e-02 - bval,  2.65404473e-02  - bval, -3.35566339e-02 - bval, 3.42296008e-02  - bval,  5.76184965e-02 - bval, -9.91353950e-02 - bval,  1.29417795e-01-bval]

x_MLE_up = [ 0.001 * 696992.0, 0.001 * 696992.0,  1.30772565e-01 + bval, -8.54004108e-02 + bval,  2.65404473e-02  + bval, -3.35566339e-02 + bval, 3.42296008e-02 + bval,  5.76184965e-02 + bval, -9.91353950e-02 + bval, 1.29417795e-01+bval ]




############# SYNTHETIC DATA ######################

# ### FOR CASES WITH TRUE PARAMETER VALUES - SYNTHETIC DATA  ###############
# ## True parameters of your model
# phiTrue = [ 430,430, 0.15, -0.1, 0.05,  -0.07, 0.035, 0.08, -0.065,0.025] 


### MLE OF YOUR PARAMETERS WHICH WILL BE USED AS PRIORS FOR TMCMC. UNIFORM PRIORS WITHIN THE BOUNDS.
# x_MLE_low = np.zeros([10])
# x_MLE_up = np.zeros([10])


# bval = 0.2

# # [ 0.15052827 -0.10283953  0.05488463 -0.07587531  0.0391744   0.07864956
# #  -0.07485274  0.04286198]

# x_MLE_low = [0,                    0,                0.15052827  - bval, -0.10283953 - bval,  0.05488463  - bval, -0.07587531 - bval, 0.0391744  - bval,  0.07864956 - bval, -0.07485274 - bval,  0.04286198-bval]

# x_MLE_up = [ 0.001 * 2794356.0, 0.001 * 2794356.0,  0.15052827 + bval, -0.10283953 + bval,  0.05488463  + bval, -0.07587531 + bval, 0.0391744  + bval,  0.07864956 + bval, -0.07485274 + bval, 0.04286198+bval ]


# [ 0.15       -0.1000742   0.06064663 -0.0891702   0.04714598  0.04359538
#   0.01      ]

# [ 0.14588876 -0.10323568  0.05907849 -0.06301169  0.02393115  0.07729624]

### FOR MAP ESTIMATE

# x_MLE_low = [ 0.14588876  - bval, -0.10323568 - bval,  0.05907849 - bval, -0.06301169 - bval, 0.02393115  - bval,  0.07729624 - bval]

# x_MLE_up = [  0.14588876  + bval, -0.10323568 +  bval,  0.05907849 + bval, -0.06301169 + bval, 0.02393115  + bval,  0.07729624 + bval]

X_low = x_MLE_low
X_up = x_MLE_up


#### CHANGE HERE 
#### LABEL VALUES TO BE INCLUDED IN PLOTS

mylabel = [r'$E_{0}$',r'$I_{0}$', r'$a_{0}$',r'$a_{1}$', r'$a_{2}$', r'$a_{3}$', r'$a_{4}$', r'$a_{5}$',r'$a_{6}$',r'$a_{7}$']

#Generates random variables for each of the parameters:
all_params = [None] * Npar #initialize list of parameters
for jj in range(0,Npar):
    pdfcur = pdfs.Uniform(lower=X_low[jj], upper=X_up[jj])
    all_params[jj] = pdfcur


####### CHANGE HERE #####################
dt = 0.1
tstart = 0
tlim = 210
t = np.arange(tstart, tlim, 1)

tmoh = np.arange(0, tlim, dt)

ndiv = 1/dt

N_city = 1



# Model parameters - Taken from Southern Ontario - COVID MBE paper
gamma_e = 1/15
gamma_i = 1/5
gamma_r = 1/11
gamma_d = 1/750

beta_e = np.zeros((len(tmoh),N_city))
beta_i = np.zeros((len(tmoh),N_city))


# Preallocate compartments
S = np.zeros((len(tmoh),N_city))
E = np.zeros((len(tmoh),N_city))
I = np.zeros((len(tmoh),N_city))
R = np.zeros((len(tmoh),N_city))
D = np.zeros((len(tmoh),N_city))
N = np.zeros((len(tmoh),N_city))

PHU_path = '/Users/sudhipv/documents/coupled_compartment/PHU_Data'
figpath = '/Users/sudhipv/documents/coupled_compartment/figs/mcmc'
datapath = '/Users/sudhipv/documents/coupled_compartment/data'
Data = np.zeros([365,4])

target_file1 = f'{PHU_path}/30-Toronto.csv'
target_file2 = f'{PHU_path}/34-York.csv'
target_file3 = f'{PHU_path}/04-Durham.csv'
target_file4 = f'{PHU_path}/22-PeelRegion.csv'

Data[:,0] = np.genfromtxt(target_file1, delimiter=',')
Data[:,1] = np.genfromtxt(target_file2, delimiter=',')
Data[:,2] = np.genfromtxt(target_file3, delimiter=',')
Data[:,3] = np.genfromtxt(target_file4, delimiter=',')

population_by_phu = np.genfromtxt(f'{PHU_path}/population_by_phu.csv', delimiter=',')

####### CHANGE HERE #####################
total = population_by_phu[3,1]

# target_file1 = './toronto_synthetic_data_75.csv'

####### CHANGE HERE #####################
#### FOR LOADING YOUR SYNTHETIC DATA

# I_synthetic = np.zeros((len(t),N_city))
# # file = np.genfromtxt(f'{datapath}/toronto_synthetic_data_noise10.csv', delimiter=',')
# file = np.genfromtxt(f'{datapath}/toronto_synthetic_data_r100.csv', delimiter=',')
# I_synthetic[:,0] = file[tstart:tlim]

#### OBSERVED MOH DATA
I_synthetic = np.zeros((len(t),N_city))
I_synthetic[:,0] =  Data[tstart:tlim,2]

t1 =  20

t2 =  35

t3 = 65

t4 = 90

t5 = 140

t6 = 180

t7 = 190


r = 100 ### number of successes

def loglikfun(param):

    ###### CHANGE HERE ###########
    E[0,0] = param[0]
    I[0,0] = param[1]
    N[0,0] = total


    ###### CHANGE HERE ###########

    R[0,0] = 0
    D[0,0] = 0
    S[0,0] = N[0,0] - E[0,0] - I[0,0] - R[0,0] - D[0,0]

####### CHANGE HERE #####################
#### USE THE TIME FOR EACH SIGMOID ACCORDING TO YOUR PHU #########


    beta_i[:,0] = param[2]  + param[3]/(1 + np.exp((t1-tmoh))) \
        +  param[4]/(1 + np.exp((t2-tmoh))) + param[5]/(1 + np.exp((t3-tmoh))) \
        + param[6]/(1 + np.exp((t4-tmoh)))  + param[7]/(1 + np.exp((t5-tmoh)))  \
        + param[8]/(1 + np.exp((t6-tmoh))) + param[9]/(1 + np.exp((t7-tmoh)))

    # + parameter_vector_in[8]/(1 + np.exp((t8-tmoh)))

    beta_e[:,0] = beta_i[:,0]

    ### Only with FoI - No mobility tensor

    FoI = np.zeros((len(tmoh),1))

    idxmoh = 1
    idxdata= 0

    loglik = 0


#### For adding the log likelihood for initial condition


    logNfac = 0.0

    if int(I_synthetic[0,0]) != 0:
        logNfac = np.sum(np.log(np.arange(0,int(I_synthetic[0,0]),1)+1)) #log factorial

    # r = (p*I[kk,0])/(1-p)
    p = r /(I[0,0] + r)
    loglik = loglik + (math.lgamma(I_synthetic[0,0]+r) - (logNfac + math.lgamma(r)) + r*np.log(p) + I_synthetic[0,0]*np.log(1-p))


    # multiplier = (1/(np.sqrt(2*np.pi)*sigma * I[0,0]))

    # err = (I_synthetic[0,0] -  I[0,0])**2

    # # log likelihood
    # loglik = loglik  + np.log(multiplier) - (err/(2*(sigma * I[0,0])**2)) 

    for kk in range(1,len(tmoh)):

        FoI[kk,0] = beta_e[kk-1,0] * (E[kk-1,0] + I[kk-1,0]) / N[kk-1,0]

        S[kk,0] = S[kk-1,0] + dt*(- FoI[kk,0] * S[kk-1,0])
        E[kk,0] = E[kk-1,0] + dt*(FoI[kk,0]*S[kk-1,0] - (gamma_i + gamma_e)*E[kk-1,0])
        I[kk,0] = I[kk-1,0] + dt*(gamma_i*E[kk-1,0] - (gamma_r + gamma_d)*I[kk-1,0])
        R[kk,0] = R[kk-1,0] + dt*(gamma_e*E[kk-1,0] + gamma_r*I[kk-1,0])
        D[kk,0] = D[kk-1,0] + dt*(gamma_d*I[kk-1,0])
        N[kk,0] = S[kk,0] +  E[kk,0] + I[kk,0] + R[kk,0]


## For collecting the model output only at data points
        if( kk%ndiv == 0 ):
            idxmoh = int(kk/ndiv)

            if(tstart != 0):
                if(idxmoh >= tstart and idxmoh < tlim):
                        
                    logNfac = 0.0

                    if int(I_synthetic[idxdata,0]) != 0:
                        logNfac = np.sum(np.log(np.arange(0,int(I_synthetic[idxdata,0]),1)+1)) #log factorial

                    # r = (p*I[kk,0])/(1-p)
                    p = r /(I[kk,0] + r)
                    loglik = loglik + (math.lgamma(I_synthetic[idxdata,0]+r) - (logNfac + math.lgamma(r)) + r*np.log(p) + I_synthetic[idxdata,0]*np.log(1-p))
                    idxdata+=1
        
            else:         

                logNfac = 0.0

                if int(I_synthetic[idxmoh,0]) != 0:
                    logNfac = np.sum(np.log(np.arange(0,int(I_synthetic[idxmoh,0]),1)+1)) #log factorial

                # r = (p*I[kk,0])/(1-p)
                p = r /(I[kk,0] + r)
                loglik = loglik + (math.lgamma(I_synthetic[idxmoh,0]+r) - (logNfac + math.lgamma(r)) + r*np.log(p) + I_synthetic[idxmoh,0]*np.log(1-p))

                if(np.isnan(loglik)):
                    print("I is", I[kk,0])
                    print("kk is", kk, idxmoh)
                    print("param,", param )
                    print("beta,", beta_i[kk,0] )
                    loglik = -1 * np.inf
                    

    return loglik


def logpriorpdf():
    '''Define Log Prior Function'''
    logprior = 0.0
    return logprior

# Log Posterior to use with TMCMC code:
def logposterior(parameter_vector_in):
    '''Define Log Posterior Function'''
    return logpriorpdf() + loglikfun(parameter_vector_in)




############ CHANGE HERE ############## 
##### REMEMBER TO CHANGE THE PATH TO FIGURES IF YOU WANT IT SAVED IN SPECIFIC LOCATIONS #########
if __name__ == '__main__': #the main part of the program.

    ####### CHANGE HERE #####################
    ### Number of samples to use at each stage
    Nsmp = 20
    import time
    start = time.time()

    Xsmp,Chain,_,comm = run_tmcmc(Nsmp,all_params,logposterior,parallel_processing,f'{datapath}/stat-file-tmcmc_durham_real.txt')

##### IF YOU WANT TO LOAD PREVIOUSLY GENERATED SAMPLES 
    # Xsmp = np.loadtxt(f'{datapath}/muVec_synthetic_case2.dat')
    # Chain = np.loadtxt(f'{datapath}/muVec_long_synthetic_case2.dat')


    end = time.time()
    print(end - start)

    Xsmp = Xsmp.T
    np.savetxt(f'{datapath}/muVec_durham_real.dat',Xsmp)
    np.savetxt(f'{datapath}/muVec_long_durham_real.dat',Chain)

    mpl.rcParams.update({'font.size':14})
    for ii in range(0,Npar):
        plt.figure(ii,figsize=(3.5, 2.8))
        plt.plot((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar),Chain[ii::Npar],'b.',markersize=2)
        #plt.plot(Xsmp[ii,:],Chain)

        ### CHANGE HERE ######
        ### Uncomment for Synthetic Data ######
        # plt.plot([0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar))[-1])],[phiTrue[ii],phiTrue[ii]],'r--',label='True')


        # plt.legend(loc='best')
        myXTicks = np.arange(0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar))[-1])+1,2)
        plt.xticks(myXTicks)
        plt.xlim([0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar)+0.0001)[-1])])
        plt.grid(True)
        #plt.xlim([0,3])
        plt.xlabel('Stage')
        plt.ylabel(mylabel[ii])
        plt.savefig(f'{figpath}/Chain'+str(ii+1)+'.eps',bbox_inches='tight')
        plt.close()

    mpl.rcParams.update({'font.size':14})
    statSmp = Xsmp.copy()
    pdfMAP = np.zeros((Npar))
    for j in range(0,Npar):
        fig = plt.figure(1+j,figsize=(3.5, 2.8))
        ax1 = fig.gca()
        xlow, xup = np.min(statSmp[j,:]),np.max(statSmp[j,:])
        Xpdf = st.kde.gaussian_kde(statSmp[j,:],bw_method = 0.3)  ## adjust kernel width
        print(Xpdf.silverman_factor())
        Xgrd = np.linspace(xlow,xup,100)
        ax1.plot(Xgrd,Xpdf(Xgrd),'b-')
        pdfmax = max(Xpdf(Xgrd))
        pdfMAP[j] = Xgrd[np.argmax(Xpdf(Xgrd))] #calculates the MAP estimate of the PDF.
        pdfStd = np.std(statSmp[j,:],0)
        pdfMean = np.mean(statSmp[j,:],0)
        pdfCOV = abs(pdfStd/pdfMean)

        ### FOR REAL DATA
        ax1.axvline(pdfMAP[j],c='r',linestyle='--', label='MAP')
        print('MAP estimate for '+mylabel[j]+': '+str(pdfMAP[j]))

        print('COV for '+mylabel[j]+': '+str(pdfCOV))
        myYlim = [0.0, 1.1*pdfmax]
        if j ==0:
             ### CHANGE HERE ######
             ### Synthetic Data - Uncomment ######
            ax1.legend(loc='upper left', numpoints = 1)
        print(myYlim)
        print('=======================')
        ### CHANGE HERE ######
        ### Synthetic Data - Uncomment ######
        # ax1.plot([phiTrue[j],phiTrue[j]],myYlim,'--r')
        ax1.set_ylabel('pdf')
        ax1.set_xlabel(mylabel[j])
        #plt.xlim([np.min(statSmp[j,:]),np.max(statSmp[j,:])])
        ax1.set_ylim(myYlim)
        ax1.set_xlim([xlow,xup])
        # plt.xlim([X_low[j],X_up[j]])
        ax1.set_yticks([])
        plt.grid(True)
        ax2 = ax1.twinx()
        Nbins = int(np.sqrt(Nsmp))
        y,x,_ = ax2.hist(statSmp[j,:],alpha=0.1,bins=Nbins) #y and x return the bin locations and number of samples for each bin, respectively
        myYlim2 = [0,1.1*y.max()]
        ax2.set_ylim(myYlim2)
        ax2.set_yticks([])

        plt.savefig(f'{figpath}/mpdf_'+str(j)+'.pdf',bbox_inches='tight')
        plt.close()

    msize = 1.2
    for i in range(0,Npar):
        for j in range(0,Npar):

            if(i != j):
                plt.figure(Npar*i+j,figsize=(3.5, 2.8))
                plt.plot(Xsmp[i,:],Xsmp[j,:],'b.',markersize = msize)

                ### CHANGE HERE ######
                ### Synthetic Data - Uncomment below lines ######
                # plt.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
                # plt.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
                # plt.xlim([phiTrue[i]-0.02,phiTrue[i]+0.02])
                # plt.ylim([phiTrue[j]-0.02,phiTrue[j]+0.02])

                plt.xlabel(mylabel[i])
                plt.ylabel(mylabel[j])

                ### REAL DATA ###
                plt.xlim([np.min(statSmp[i,:]),np.max(statSmp[i,:])])
                plt.ylim([np.min(statSmp[j,:]),np.max(statSmp[j,:])])

                plt.grid(True)
                plt.savefig(f'{figpath}/Jsmpls_'+str(i+1)+str(j+1)+'.eps',bbox_inches='tight')
                plt.close()

    msize = 1.2
    for i in range(0,Npar):
        for j in range(0,Npar):

            if(i != j):

                fig = plt.figure(Npar*i+j,figsize=(3.5, 2.8))
                xmin = np.min(statSmp[i,:])
                xmax = np.max(statSmp[i,:])
                ymin = np.min(statSmp[j,:])
                ymax = np.max(statSmp[j,:])
                x = Xsmp[i,:].T
                y = Xsmp[j,:].T
                xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])
                kernel = st.gaussian_kde(values,bw_method = 1)
                f = np.reshape(kernel(positions).T, xx.shape)
                ax = fig.gca()
                # Contourf plot
                cfset = ax.contourf(xx, yy, f, 15,cmap='Blues')
                ## Or kernel density estimate plot instead of the contourf plot
                #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
                # Contour plot
                #cset = ax.contour(xx, yy, f, colors='k')
                # Label plot
                #ax.clabel(cset, inline=1, fontsize=10)

                ### CHANGE HERE ######
                ### Synthetic Data - Uncomment below line ######
                # plt.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
                # plt.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
                # plt.xlim([phiTrue[i]-0.02,phiTrue[i]+0.02])
                # plt.ylim([phiTrue[j]-0.02,phiTrue[j]+0.02])

                plt.xlabel(mylabel[i])
                plt.ylabel(mylabel[j])
                ### REAL DATA ###
                plt.xlim([np.min(statSmp[i,:]),np.max(statSmp[i,:])])
                plt.ylim([np.min(statSmp[j,:]),np.max(statSmp[j,:])])

                plt.grid(True)
                plt.savefig(f'{figpath}/jpdf_'+str(i+1)+str(j+1)+'.eps',bbox_inches='tight')
                plt.close()

    # kdeMCMC= st.gaussian_kde(statSmp,bw_method = 0.1)
    # SigMat = kdeMCMC.covariance
    # np.savetxt('SigMat.dat',SigMat)
