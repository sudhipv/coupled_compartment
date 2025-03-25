
#### Code to sample posterior pdf for 2 parameters of coupled PHUs with mobility.


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

parallel_processing = 'multiprocessing' #'multiprocessing','mpi'

# print("code started")
Npar = 20 # number of unknown parameters

x_MLE_low = np.zeros([20])
x_MLE_up = np.zeros([20])

bval = 0.2

# # MLE values

########### REAL  DATA ##################

# [ 0.15530582 -0.11440652  0.068809   -0.06500159  0.02018632  0.08993903
#  -0.03972402 -0.01715194  0.1114486  -0.06963404  0.00147462 -0.0326881
#   0.03587561  0.03286086 -0.11130769  0.11692651]


# x_MLE_low = [ 0, 0 , 0.15530582-bval, -0.11440652-bval,  0.068809-bval,   -0.06500159-bval,  0.02018632-bval,  0.08993903-bval, \
#  -0.03972402-bval, -0.01715194-bval, 0, 0,  0.1114486-bval,  -0.06963404-bval,  0.00147462-bval, -0.0326881-bval, \
#   0.03587561-bval,  0.03286086-bval, -0.11130769-bval,  0.11692651-bval]

# x_MLE_up = [0.001 * 2794356.0, 0.001 * 2794356.0,  0.15530582+bval, -0.11440652+bval,  0.068809+bval,   -0.06500159+bval,  0.02018632+bval,  0.08993903+bval, \
#  -0.03972402+bval, -0.01715194+bval, 0.001 * 696992.0, 0.001 * 696992.0,  0.1114486+bval,  -0.06963404+bval,  0.00147462+bval, -0.0326881+bval, \
#   0.03587561+bval,  0.03286086+bval, -0.11130769+bval,  0.11692651+bval]


########### SYNTHETIC DATA ##################

a01 =   0.15
a11 =  -0.1
a21 =   0.05
a31 =   -0.07
a41=  0.035
a51 =  0.08
a61 =  -0.065
a71 =  0.025
# a81 =  -0.015

a02 =   0.14
a12 =  -0.115
a22 =   0.06
a32 =   -0.05
a42 =  0.035
a52 =  0.06
a62 =  -0.075
a72 =  0.06
# a82 =  -0.025

# 210 cutoff
phiTrue = [430,430 ,a01,a11,a21,a31,a41,a51,a61,a71, 94,94, a02,a12,a22,a32,a42,a52,a62,a72]

#### MLE estimates
# [ 0.15062696 -0.10442009  0.05794303 -0.07842871  0.03928469  0.08009362
#  -0.07175242  0.03570238  0.14112474 -0.114738    0.05581739 -0.04533642
#   0.03257459  0.06379187 -0.07886659  0.06675659]



x_MLE_low = [ 0, 0 , 0.15062696 - bval , -0.10442009 - bval, 0.05794303 - bval, -0.07842871 - bval, 0.03928469 - bval, 0.08009362 - bval, -0.07175242 - bval, 0.03570238 - bval, \
              0, 0, 0.14112474 -bval,  -0.114738 - bval, 0.05581739 - bval,  -0.04533642 - bval, 0.03257459 - bval,0.06379187 - bval,  -0.07886659 - bval, 0.06675659 - bval]


x_MLE_up = [ 0.001 * 2794356.0, 0.001 * 2794356.0 , 0.15062696 + bval , -0.10442009 + bval, 0.05794303 + bval, -0.07842871 + bval,0.03928469 + bval,  0.08009362 + bval, -0.07175242 + bval, 0.03570238 + bval, \
             0.001 * 696992.0, 0.001 * 696992.0, 0.14112474 + bval,  -0.114738 + bval, 0.05581739 + bval,  -0.04533642 + bval, 0.03257459 + bval,0.06379187 + bval,  -0.07886659 + bval, 0.06675659 + bval]


## 160 cut-off
# phiTrue = [430,430 ,a01,a11,a21,a31,a41,a51, 94,94, a02,a12,a22,a32,a42,a52]

# x_MLE_low = [ 0, 0 , 0.15062696 - bval , -0.10442009 - bval, 0.05794303 - bval, -0.07842871 - bval, 0.03928469 - bval,  0.08009362 - bval, \
#               0, 0, 0.14112474 -bval,  -0.114738 - bval, 0.05581739 - bval,  -0.04533642 - bval, 0.03257459 - bval, 0.06379187 - bval]


# x_MLE_up = [ 0.001 * 2794356.0, 0.001 * 2794356.0 , 0.15062696 + bval , -0.10442009 + bval, 0.05794303 + bval, -0.07842871 + bval, 0.03928469 + bval,  0.08009362 + bval, \
#             0.001 * 696992.0, 0.001 * 696992.0, 0.14112474 + bval,  -0.114738 + bval, 0.05581739 + bval,  -0.04533642 + bval, 0.03257459 + bval, 0.06379187 + bval]

############################# ############################# #############################

X_low = x_MLE_low
X_up = x_MLE_up

mylabel = [r'$E_{0}^{1}$',r'$I_{0}^{1}$', r'$a_{0}^{1}$',r'$a_{1}^{1}$', r'$a_{2}^{1}$',r'$a_{3}^{1}$', r'$a_{4}^{1}$',r'$a_{5}^{1}$',r'$a_{6}^{1}$',r'$a_{7}^{1}$',\
           r'$E_{0}^{2}$',r'$I_{0}^{2}$', r'$a_{0}^{2}$',r'$a_{1}^{2}$', r'$a_{2}^{2}$',r'$a_{3}^{2}$', r'$a_{4}^{2}$',r'$a_{5}^{2}$',r'$a_{6}^{2}$',r'$a_{7}^{2}$']

#Generates random variables for each of the parameters:
all_params = [None] * Npar #initialize list of parameters
for jj in range(0,Npar):
    pdfcur = pdfs.Uniform(lower=X_low[jj], upper=X_up[jj])
    all_params[jj] = pdfcur



############ Likelihood needed stuff


dt = 0.1
tstart = 0
tlim = 210
t = np.arange(tstart, tlim, 1)

tmoh = np.arange(tstart, tlim, dt)

tmobility = np.arange(tstart, 272, dt)

ndiv = 1/dt

N_city = 2
# Model parameters - Taken from Southern Ontario - COVID MBE paper
gamma_e = 1/15
gamma_i = 1/5
gamma_r = 1/11
gamma_d = 1/750

beta_e = np.zeros((len(tmoh),N_city))
beta_i = np.zeros((len(tmoh),N_city))

# t1 =  20

# Preallocate compartments
S = np.zeros((len(tmoh),N_city))
E = np.zeros((len(tmoh),N_city))
I = np.zeros((len(tmoh),N_city))
R = np.zeros((len(tmoh),N_city))
D = np.zeros((len(tmoh),N_city))
N = np.zeros((len(tmoh),N_city))



PHU_path = '/Users/sudhipv/documents/coupled_compartment/PHU_Data'
figpath = '/Users/sudhipv/documents/coupled_compartment/figs'
datapath = '/Users/sudhipv/documents/coupled_compartment/data'
mobpath = '/Users/sudhipv/documents/coupled_compartment/mobility_tensor'

# PHU_path = './../../PHU_Data'
# figpath = './figs'
# datapath = './../../data'
# mobpath = './../../mobility_tensor'

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



# Stratification Tensor M
M = np.zeros((4,N_city,N_city,len(tmobility)))

# print("shape of M is", np.shape(M))


#####################################################################################


#  Below part of code is not well written - It uses the weekly data to generate the daily values
# of mobility tensor and further saves in each time step.

# Load the Mobility tensor values based on the Flow matrix

# Week number 13 (starting from 0) is April 6th.
# Week number 37 is Sep 21st.

# os.chdir('/Users/sudhipv/documents/coupledode/codes/mobility_tensor/stochastic_Tor_Durham')
Mtensor = np.zeros([52,2,2])
current_directory = os.getcwd()
# print(current_directory)

directory_path = f'{mobpath}/stochastic_Tor_Durham/'
filesmobility = os.listdir(directory_path)

# print(files)

for ii in range(52):

    target_name_part = "Stochastic_matrix"+str(ii)+".dat"

    for file in filesmobility:
        if target_name_part == file:
            target_file = file
#           print(target_name_part)
#           print(target_file)
            break
    else:
        raise FileNotFoundError(f"Could not find any file containing '{target_name_part}'.")

    Mtensor[ii,:,:] = np.genfromtxt(directory_path + target_file, delimiter=' ')


# print(np.shape(Mtensor))


mrange = int((tlim-tstart)/7) + 1

# print("m range is",mrange)


for w in range(mrange): # 39

    sw = 7*w
    # print("sw is", sw)

    if(w != mrange-1):
        tsw = np.linspace(sw, sw+6, int((1/dt)*6+1))
    else:
        tsw = np.linspace(sw, sw+5, int((1/dt)*5+1))


    # print("length of tsw is", len(tsw))
    for swk in range(len(tsw)):
        mindex = int((1/dt) * sw + swk)
        # print("mindex is", mindex)
        M[0,:,:,mindex] = Mtensor[13+w,0:N_city,0:N_city]
        M[1,:,:,mindex] = Mtensor[13+w,0:N_city,0:N_city]
        M[2,:,:,mindex] = Mtensor[13+w,0:N_city,0:N_city]
        M[3,:,:,mindex] = Mtensor[13+w,0:N_city,0:N_city]


    tsw2 = np.linspace(sw+6+dt, sw+7, int((1/dt)*1))
    # print("tsw2 is", tsw2)
    for swk2 in range(1,len(tsw2)):
        mindex2 = mindex + swk2
        # print(mindex2)
        M[0,:,:,mindex2] = Mtensor[13+w,0:N_city,0:N_city]
        M[1,:,:,mindex2] = Mtensor[13+w,0:N_city,0:N_city]
        M[2,:,:,mindex2] = Mtensor[13+w,0:N_city,0:N_city]
        M[3,:,:,mindex2] = Mtensor[13+w,0:N_city,0:N_city]


# print("mobility matrix created")

##### Force of infection , Lambda
L_Force = np.zeros((len(tmoh),N_city))


total = np.zeros((N_city))

total[0] = population_by_phu[29,1]

total[1] = population_by_phu[3,1]


####### CHANGE HERE #####################
#### FOR LOADING YOUR SYNTHETIC DATA


I_synthetic = np.zeros((len(t),N_city))
# file = np.genfromtxt(f'{datapath}/toronto_synthetic_data_noise10.csv', delimiter=',')
file = np.genfromtxt(f'{datapath}/coupled_synth_data_r100.csv', delimiter=',')
I_synthetic[:,0] = file[tstart:tlim,0]
I_synthetic[:,1] = file[tstart:tlim,1]

#### OBSERVED MOH DATA
# I_synthetic = np.zeros((len(t),N_city))
# I_synthetic[:,0] =  Data[tstart:tlim,0]
# I_synthetic[:,1] =  Data[tstart:tlim,2]


r = 100 ### number of successes


def loglikfun(param):

    # Initial Conditions

    E[0,0] = param[0]
    I[0,0] = param[1]
    R[0,0] = 0
    D[0,0] = 0
    N[0,0] = total[0]
    S[0,0] = N[0,0] - E[0,0] - I[0,0] - R[0,0] - D[0,0]

    # Toronto
    t1 =  20
    t2 =  35
    t3 = 60
    t4 = 80
    t5 = 140
    t6 = 180
    t7 = 190
    t8 = 230

    beta_i[:,0] = param[2] + param[3]/(1 + np.exp((t1-tmoh))) +  param[4]/(1 + np.exp((t2-tmoh))) + param[5]/(1 + np.exp((t3-tmoh))) + param[6]/(1 + np.exp((t4-tmoh))) + param[7]/(1 + np.exp((t5-tmoh)))
    + param[8]/(1 + np.exp((t6-tmoh))) + param[9]/(1 + np.exp((t7-tmoh)))

    # + param[8]/(1 + np.exp((t8-tmoh)))

    beta_e[:,0] = beta_i[:,0]


    E[0,1] = param[10]
    I[0,1] = param[11]
    R[0,1] = 0
    D[0,1] = 0
    N[0,1] = total[1]
    S[0,1] = N[0,1] - E[0,1] - I[0,1] - R[0,1] - D[0,1]

    # Durham
    t1 =  20
    t2 =  35
    t3 = 65
    t4 = 90
    t5 = 140
    t6 = 180
    t7 = 190
    t8 = 250

    # print((Npar/2)+ 0)

    beta_i[:,1] = param[12]  + param[13]/(1 + np.exp((t1-tmoh))) +  param[14]/(1 + np.exp((t2-tmoh))) + param[15]/(1 + np.exp((t3-tmoh))) + param[16]/(1 + np.exp((t4-tmoh))) \
                  + param[17]/(1 + np.exp((t5-tmoh))) + param[18]/(1 + np.exp((t6-tmoh))) + param[19]/(1 + np.exp((t7-tmoh)))

                 # + param[17]/(1 + np.exp((t8-tmoh)))

    beta_e[:,1] = beta_i[:,1]


    idxmoh = 1
    idxdata= 0

    loglik = 0



    logNfac = 0.0


    for gg in range(0,N_city):

        if int(I_synthetic[0,gg]) != 0:
            logNfac = np.sum(np.log(np.arange(0,int(I_synthetic[0,gg]),1)+1)) #log factorial

        # r = (p*I[kk,0])/(1-p)
        p = r /(I[0,gg] + r)
        loglik = loglik + (math.lgamma(I_synthetic[0,gg]+r) - (logNfac + math.lgamma(r)) + r*np.log(p) + I_synthetic[0,gg]*np.log(1-p))


    for kk in range(1,len(tmoh)):

        for gg in range(0,N_city):

            L_sum = 0
            for ll in range(0,N_city):

                Nlm = 0
                L_cityinf = 0

                for mm in range(0,N_city):
                    Nlm =  Nlm + M[0,mm,ll,kk-1] * S[kk-1,mm] +  M[1,mm,ll,kk-1] * E[kk-1,mm] + M[2,mm,ll,kk-1] * I[kk-1,mm] + M[3,mm,ll,kk-1] * R[kk-1,mm]

                    L_cityinf = L_cityinf + (beta_e[kk-1,ll] * M[1,mm,ll,kk-1] * E[kk-1,mm] + beta_i[kk-1,ll] * M[2,mm,ll,kk-1] * I[kk-1,mm])


                L_sum = L_sum + (M[0,gg,ll,kk-1] * L_cityinf)/Nlm


            L_Force[kk-1,gg] = L_sum


            S[kk,gg] = S[kk-1,gg] - dt*(L_Force[kk-1,gg]*S[kk-1,gg])
            E[kk,gg] = E[kk-1,gg] + dt*(L_Force[kk-1,gg]*S[kk-1,gg] - (gamma_i + gamma_e)*E[kk-1,gg])
            I[kk,gg] = I[kk-1,gg] + dt*(gamma_i*E[kk-1,gg] - (gamma_r + gamma_d)*I[kk-1,gg])
            R[kk,gg] = R[kk-1,gg] + dt*(gamma_e*E[kk-1,gg] + gamma_r*I[kk-1,gg])
            D[kk,gg] = D[kk-1,gg] + dt*(gamma_d*I[kk-1,gg])
            N[kk,gg] = S[kk,gg] +  E[kk,gg] + I[kk,gg] + R[kk,gg]



            # For collecting the model output only at data points
            if( kk%ndiv == 0 ):
                idxmoh = int(kk/ndiv)

                if(tstart != 0):
                    if(idxmoh >= tstart and idxmoh < tlim):

                        logNfac = 0.0

                        if int(I_synthetic[idxdata,gg]) != 0:
                            logNfac = np.sum(np.log(np.arange(0,int(I_synthetic[idxdata,gg]),1)+1)) #log factorial

                        # r = (p*I[kk,0])/(1-p)
                        p = r /(I[kk,gg] + r)
                        loglik = loglik + (math.lgamma(I_synthetic[idxdata,gg]+r) - (logNfac + math.lgamma(r)) + r*np.log(p) + I_synthetic[idxdata,gg]*np.log(1-p))
                        idxdata+=1

                else:

                    logNfac = 0.0

                    if int(I_synthetic[idxmoh,gg]) != 0:
                        logNfac = np.sum(np.log(np.arange(0,int(I_synthetic[idxmoh,gg]),1)+1)) #log factorial

                    # r = (p*I[kk,0])/(1-p)
                    p = r /(I[kk,gg] + r)
                    loglik = loglik + (math.lgamma(I_synthetic[idxmoh,gg]+r) - (logNfac + math.lgamma(r)) + r*np.log(p) + I_synthetic[idxmoh,gg]*np.log(1-p))

                    if(np.isnan(loglik)):
                        # print("I is", I[kk,gg])
                        # print("kk is", kk, idxmoh)
                        # print("param,", param )
                        # print("beta,", beta_i[kk,gg] )
                        loglik = -1 * np.inf


    return loglik



def logpriorpdf():
    '''Define Log Prior Function'''
    logprior = 0.0
    return logprior

#Log posterior to use with Rimple's code
# def logposterior(parameter_vector_in):
#     '''Define Log Posterior Function'''
#     return logpriorpdf() + loglikfun(parameter_vector_in)

# Log Posterior to use with TMCMC code:
def logposterior(parameter_vector_in):
    '''Define Log Posterior Function'''
    return logpriorpdf() + loglikfun(parameter_vector_in)

if __name__ == '__main__': #the main part of the program.
    Nsmp = 40
    # loglikfun([0.15,0.14])
    import time
    start = time.time()
    #Xsmp,Chain,LLsmp, Evid, tmcmcFac  = tmcmc_alpha.tmcmc(logposterior,Npar,X_low,X_up,Nsmp)
    # print("code inside main")

    # Xsmp,Chain,_,comm = run_tmcmc(Nsmp,all_params,logposterior,parallel_processing,f'{datapath}/stat-file-tmcmc_coupled_synth_210.txt')

    Xsmp = np.loadtxt(f'{datapath}/muVec_coupled_synth_210.dat')
    Chain = np.loadtxt(f'{datapath}/muVec_long_coupled_synth_210.dat')

    # if parallel_processing == 'mpi':
    #     comm.Abort(0)
    end = time.time()
    print(end - start)

    # Xsmp = Xsmp.T
    # np.savetxt(f'{datapath}/muVec_coupled_synth_210.dat',Xsmp)
    # np.savetxt(f'{datapath}/muVec_long_coupled_synth_210.dat',Chain)

    mpl.rcParams.update({'font.size':14})
    for ii in range(0,Npar):
        plt.figure(ii,figsize=(3.5, 2.8))
        plt.plot((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar),Chain[ii::Npar],'b.',markersize=2)
        #plt.plot(Xsmp[ii,:],Chain)

        ### CHANGE HERE ######
        ### Uncomment for Synthetic Data ######
        plt.plot([0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar))[-1])],[phiTrue[ii],phiTrue[ii]],'r--',label='True')


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
        # ax1.axvline(pdfMAP[j],c='r',linestyle='--', label='MAP')
        # print('MAP estimate for '+mylabel[j]+': '+str(pdfMAP[j]))

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
        ax1.plot([phiTrue[j],phiTrue[j]],myYlim,'--r')
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
                plt.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
                plt.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
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
                plt.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
                plt.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
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
