
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

import tmcmc_alpha
# From opensource repo:
from tmcmc_mod import pdfs
from tmcmc_mod.tmcmc_mod import run_tmcmc

np.random.seed(106)  # fixing the random seed

parallel_processing = 'mpi' #'multiprocessing','mpi'

print("code started")
Npar = 16 # number of unknown parameters

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



# MLE values
# xk = [ 0.15478438 -0.10441809  0.04505573 -0.05512462  0.01869748  0.08734956  -0.07633955  0.0593924   
# 0.12858173 -0.09107275  0.05155948 -0.07005601 0.05614092  0.06530606 -0.08574362  0.12503362]



phiTrue = [a01,a11,a21,a31,a41,a51,a61,a71, a02,a12,a22,a32,a42,a52,a62,a72]


x_MLE_low = np.zeros([16])
x_MLE_up = np.zeros([16])

bval = 0.05

x_MLE_low = [ 0.15478438 - bval , -0.10441809 - bval, 0.04505573 - bval, -0.05512462 - bval, \
            0.01869748 - bval,  0.08734956 - bval, -0.07633955 - bval, 0.0593924 - bval, \
            0.12858173 -bval,  -0.09107275 - bval, 0.05155948 - bval,  -0.07005601 - bval, 0.05614092 - bval,
            0.06530606 - bval,  -0.08574362 - bval, 0.12503362 - bval]


x_MLE_up = [ 0.15478438 + bval , -0.10441809 + bval, 0.04505573 + bval, -0.05512462 + bval, \
            0.01869748 + bval,  0.08734956 + bval, -0.07633955 + bval, 0.0593924 + bval, \
            0.12858173 + bval,  -0.09107275 + bval, 0.05155948 + bval,  -0.07005601 + bval, 0.05614092 + bval,
            0.06530606 + bval,  -0.08574362 + bval, 0.12503362 + bval]

X_low = x_MLE_low
X_up = x_MLE_up

mylabel = [r'$a_{0}^{1}$',r'$a_{1}^{1}$', r'$a_{2}^{1}$',r'$a_{3}^{1}$', r'$a_{4}^{1}$',r'$a_{5}^{1}$', r'$a_{6}^{1}$', r'$a_{7}^{1}$', r'$a_{0}^{2}$',r'$a_{1}^{2}$', r'$a_{2}^{2}$',r'$a_{3}^{2}$', \
           r'$a_{4}^{2}$',r'$a_{5}^{2}$', r'$a_{6}^{2}$',r'$a_{7}^{2}$']

#Generates random variables for each of the parameters:
all_params = [None] * Npar #initialize list of parameters
for jj in range(0,Npar):
    pdfcur = pdfs.Uniform(lower=X_low[jj], upper=X_up[jj])
    all_params[jj] = pdfcur



############ Likelihood needed stuff


dt = 0.1
tstart = 0
tlim = 200
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

Data = np.zeros([365,4])

target_file1 = './phudata/30-Toronto.csv'
target_file2 = './phudata/34-York.csv'
target_file3 = './phudata/04-Durham.csv'
target_file4 = './phudata/22-PeelRegion.csv'

Data[:,0] = np.genfromtxt(target_file1, delimiter=',')
Data[:,1] = np.genfromtxt(target_file2, delimiter=',')
Data[:,2] = np.genfromtxt(target_file3, delimiter=',')
Data[:,3] = np.genfromtxt(target_file4, delimiter=',')

population_by_phu = np.genfromtxt('./phudata/population_by_phu.csv', delimiter=',')



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

directory_path = './mobility_tensor/stochastic_Tor_Durham/'
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


print("mobility matrix created")

# mpl.rcParams.update({'font.size':14})
# for i in range(N_city):
#     for j in range(N_city):
#         plt.figure(i+j,figsize=(10,5))
#         plt.plot(tmoh, M[0,i,j,:], label=f'{i,j}')
#         plt.legend(loc='best')
#         plt.xlabel('Time (in days from April 2020)')
#         plt.xlim([0,tlim])
#         plt.ylabel('$M_{ij}$')
#         plt.grid()
#         plt.show()



##### Force of infection , Lambda
L_Force = np.zeros((len(tmoh),N_city))


total = np.zeros((N_city))

total[0] = population_by_phu[29,1]

total[1] = population_by_phu[3,1]


# Initial Conditions

E[0,0] = Data[0,0]
I[0,0] = Data[0,0]
R[0,0] = 0
D[0,0] = 0
N[0,0] = total[0]
S[0,0] = N[0,0] - E[0,0] - I[0,0] - R[0,0] - D[0,0]

E[0,1] = Data[0,2]
I[0,1] = Data[0,2]
R[0,1] = 0
D[0,1] = 0
N[0,1] = total[1]
S[0,1] = N[0,1] - E[0,1] - I[0,1] - R[0,1] - D[0,1]


I_retrived = np.zeros((272,N_city))

I_synthetic = np.zeros((len(t),N_city))


I_synthetic[0,0] = I[0,0]
I_synthetic[0,1] = I[0,1]


mu = np.zeros((N_city))

sigma = np.zeros((N_city))


target_file1 = './data/toronto_2phu_jedmobility.csv'

I_retrived[:,0] = np.genfromtxt(target_file1, delimiter=',')


target_file2 =  './data/durham_2phu_jedmobility.csv'

I_retrived[:,1] = np.genfromtxt(target_file2, delimiter=',')


I_synthetic[:,0] =  I_retrived[tstart:tlim,0]

I_synthetic[:,1] =  I_retrived[tstart:tlim,1]

    # print(np.shape(I_synthetic))




def loglikfun(param):

    # print("parameters inside log-likelihood is", param)
    # '''Define Log Liklihood function'''

    # Toronto
    t1 =  20
    t2 =  35
    t3 = 60
    t4 = 80
    t5 = 140
    t6 = 180
    t7 = 190
    t8 = 230

    beta_i[:,0] = param[0] + param[1]/(1 + np.exp((t1-tmoh))) +  param[2]/(1 + np.exp((t2-tmoh))) + param[3]/(1 + np.exp((t3-tmoh))) + param[4]/(1 + np.exp((t4-tmoh))) + param[5]/(1 + np.exp((t5-tmoh))) \
                    + param[6]/(1 + np.exp((t6-tmoh))) + param[7]/(1 + np.exp((t7-tmoh)))

    # + param[8]/(1 + np.exp((t8-tmoh)))

    beta_e[:,0] = beta_i[:,0]


    # Durham
    t1 =  20
    t2 =  35
    t3 = 65
    t4 = 90
    t5 = 140
    t6 = 180
    t7 = 190
    t8 = 250

    print((Npar/2)+ 0)

    beta_i[:,1] = param[8]  + param[9]/(1 + np.exp((t1-tmoh))) +  param[10]/(1 + np.exp((t2-tmoh))) + param[11]/(1 + np.exp((t3-tmoh))) + param[12]/(1 + np.exp((t4-tmoh))) + param[13]/(1 + np.exp((t5-tmoh))) \
                 + param[14]/(1 + np.exp((t6-tmoh))) + param[15]/(1 + np.exp((t7-tmoh)))

                 # + param[17]/(1 + np.exp((t8-tmoh)))

    beta_e[:,1] = beta_i[:,1]


    Lsum = 0

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


            ## For collecting the model output only at data points

            if( kk%ndiv == 0):

                idxmoh = int(kk/ndiv)

            ### Likelihood computation

                multiplier = (1/(np.sqrt(2*np.pi)*sigma[gg]))

                err = (I_synthetic[idxmoh,gg] - I[kk,gg] - mu[gg])**2

            # log likelihood
                Lsum = Lsum  + np.log(multiplier) - (err/(2*sigma[gg]**2))

    # print('Lsum is', Lsum)
    return Lsum
    # return np.log((0.5*rv1.pdf(parameter_vector_in)+0.5*rv2.pdf(parameter_vector_in)))



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
    Nsmp = 4000
    # loglikfun([0.15,0.14])
    import time
    start = time.time()
    #Xsmp,Chain,LLsmp, Evid, tmcmcFac  = tmcmc_alpha.tmcmc(logposterior,Npar,X_low,X_up,Nsmp)
    print("code inside main")

    Xsmp,Chain,_,comm = run_tmcmc(Nsmp,all_params,logposterior,parallel_processing,'./stat-file-tmcmc.txt')

    # Xsmp = np.loadtxt('/Users/sudhipv/documents/coupledode/codes/Inference/parallel_TMC/singlephu_2param/muVec.dat')
    # Chain = np.loadtxt('/Users/sudhipv/documents/coupledode/codes/Inference/parallel_TMC/singlephu_2param/muVec_long.dat')

    # if parallel_processing == 'mpi':
    #     comm.Abort(0)
    end = time.time()
    print(end - start)

    Xsmp = Xsmp.T
    np.savetxt('./muVec.dat',Xsmp)
    np.savetxt('./muVec_long.dat',Chain)

    mpl.rcParams.update({'font.size':14})
    for ii in range(0,Npar):
        plt.figure(ii,figsize=(3.5, 2.8))
        plt.plot((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar),Chain[ii::Npar],'b.',markersize=2)
        #plt.plot(Xsmp[ii,:],Chain)
        plt.plot([0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar))[-1])],[phiTrue[ii],phiTrue[ii]],'r--',label='True')
        if ii == 0:
            plt.legend(loc='upper right')
        myXTicks = np.arange(0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar))[-1])+1,4)
        plt.xticks(myXTicks)
        plt.xlim([0,math.ceil(((1/(Nsmp*Npar))*np.arange(0,len(Chain),Npar)+0.0001)[-1])])
        plt.grid(True)
        #plt.xlim([0,3])
        plt.xlabel('Stage')
        plt.ylabel(mylabel[ii])
        plt.savefig('./figs/Chain'+str(ii+1)+'.eps',bbox_inches='tight')
        plt.close()

    mpl.rcParams.update({'font.size':14})
    statSmp = Xsmp.copy()
    pdfMAP = np.zeros((Npar))
    for j in range(0,Npar):
        fig = plt.figure(1+j,figsize=(3.5, 2.8))
        ax1 = fig.gca()
        xlow, xup = np.min(statSmp[j,:]),np.max(statSmp[j,:])
        #Xpdf = st.kde.gaussian_kde(statSmp[j,:])
        Xpdf = st.kde.gaussian_kde(statSmp[j,:],bw_method = 0.3)  ## adjust kernel width
        print(Xpdf.silverman_factor())
        # Xgrd = np.linspace(np.min(statSmp[j,:]),np.max(statSmp[j,:]))
        # Xgrd = np.linspace(X_low[j],X_up[j],100)
        Xgrd = np.linspace(xlow,xup,100)
        ax1.plot(Xgrd,Xpdf(Xgrd),'b-')
        ax1.plot()
        pdfmax = max(Xpdf(Xgrd))
        pdfMAP[j] = Xgrd[np.argmax(Xpdf(Xgrd))] #calculates the MAP estimate of the PDF.
        pdfStd = np.std(statSmp[j,:],0)
        pdfMean = np.mean(statSmp[j,:],0)
        pdfCOV = abs(pdfStd/pdfMean)
        print('MAP estimate for '+mylabel[j]+': '+str(pdfMAP[j]))
        print('COV for '+mylabel[j]+': '+str(pdfCOV))
        myYlim = [0.0, 1.1*pdfmax]
        if j ==0:
            ax1.plot([phiTrue[j],phiTrue[j]],myYlim,'--r',label='True')
            ax1.legend(loc='upper left', numpoints = 1)
        print(myYlim)
        print('=======================')
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

        plt.savefig('./figs/mpdf_'+str(j)+'.pdf',bbox_inches='tight')
        plt.close()



    ###################### ###################### ###################### ###################### ######################

    ## Jointwise Liklihood Function plots with Samples (for cases where Npar >= 2) (David Clarabut)
    # nl=200
    # compgrd = 1 #options: 1 to compute the grid, 2 to load the grid from a file, all other numbers to bypass this opperation.

    # if (compgrd == 1) and (Npar>=2):
    #     # Initializtaions (full set of plots):
    #     logLik = np.zeros([nl,nl,int(((Npar**2)-Npar)/2)]) #divide the last dimension by 2 for partial outputs
    #     cur_val = np.zeros([Npar]) #the current value of w to use in computing the objective function
    #     p1 = np.zeros([nl,nl,int(((Npar**2)-Npar)/2)]) #divide the last dimension by 2 for partial outputs
    #     p2 = np.zeros([nl,nl,int(((Npar**2)-Npar)/2)]) #divide the last dimension by 2 for partial outputs
    #     countdim=0

    #     # Computations:

    #     for i in range(0,Npar):
    #         for j in range(i+1,Npar): #i+1
    #             if i!=j:
    #                 print('Now Running Log Likelihood for '+str(mylabel[i])+' and '+str(mylabel[j]))
    #                 xmin = np.min(statSmp[i,:])
    #                 xmax = np.max(statSmp[i,:])
    #                 ymin = np.min(statSmp[j,:])
    #                 ymax = np.max(statSmp[j,:])
    #                 xx = np.linspace(xmin,xmax,nl)
    #                 yy = np.linspace(ymin,ymax,nl)
    #                 [p1[:,:,countdim],p2[:,:,countdim]] = np.meshgrid(xx,yy)
    #                 for ii in range(0,nl): #i are the x-axis parameters
    #                     for jj in range(0,nl): #j are the y-axis parameters
    #                         for kk in range(0,Npar): #find the curent value that should be input to the liklihood function. All pther parameters are set to their MAP estimates.
    #                             if (kk!=i) and (kk!=j): #check to see if m is in the dimensions in the current objective function grid itteration.
    #                                 cur_val[kk] = phiTrue[kk] #X_low[kk] #X_up[kk] #phiTrue[kk]
    #                             elif (kk==i):
    #                                 cur_val[i] = p1[ii,jj,countdim] #set the current dimension to be x
    #                             elif (kk==j):
    #                                 cur_val[j] = p2[ii,jj,countdim] #set the current dimension to be y
    #                         logLik[ii,jj,countdim] = loglikfun(cur_val)# + logpriorpdf(cur_val)
    #                         print(mylabel[i]+'= '+str(cur_val[i])+' '+mylabel[j]+'= '+str(cur_val[j])+' Loglikelihood: '+str(logLik[ii,jj,countdim]))
    #                 countdim=countdim+1
    #     # Write the array to disk, also indicates the array shape (needed to import the data later) and where the slices all start.
    #     with open('/Users/sudhipv/documents/coupledode/codes/Inference/parallel_TMC/mobility/JlogLikFun.dat', 'w') as outfile:
    #         outfile.write('# Array shape: {0}\n'.format(logLik.shape))
    #         for data_slice in logLik:
    #             np.savetxt(outfile, data_slice, fmt='%-7.2f')
    #             outfile.write('# New slice\n')
    # elif (compgrd == 2) and (Npar >= 2):
    #     # Read the array from disk
    #     logLik = np.loadtxt('/Users/sudhipv/documents/coupledode/codes/Inference/parallel_TMC/mobility/JLogLikFun.dat') #note that returns a 2-D array. I will reshare the data to the correct size on the line below.
    #     logLik = logLik.reshape((nl,nl,int(((Npar**2)-Npar)/2))) #reformat the shpe of the tensor.
    #     countdim = 0
    #     p1 = np.zeros([nl,nl,int(((Npar**2)-Npar)/2)]) #divide the last dimension by 2 for partial outputs
    #     p2 = np.zeros([nl,nl,int(((Npar**2)-Npar)/2)]) #divide the last dimension by 2 for partial outputs
    #     for i in range(0,Npar):
    #         for j in range(i+1,Npar): #i+1
    #             xmin = np.min(statSmp[i,:])
    #             xmax = np.max(statSmp[i,:])
    #             ymin = np.min(statSmp[j,:])
    #             ymax = np.max(statSmp[j,:])


    #             # xmin = X_low[i]
    #             # xmax = X_up[i]
    #             # ymin = X_low[j]
    #             # ymax = X_up[j]

    #             xx = np.linspace(xmin,xmax,nl)
    #             yy = np.linspace(ymin,ymax,nl)
    #             [p1[:,:,countdim],p2[:,:,countdim]] = np.meshgrid(xx,yy)
    #             countdim=0

    # if ((compgrd == 1) or (compgrd == 2)) and (Npar >= 2):
    #     msize = 1.2
    #     mpl.rcParams.update({'font.size':12}) #set the font size on the plots to be 10.
    #     countdim = 0 #used to track how many itmes we've been through the loop
    #     for i in range(0,Npar):
    #         for j in range(i+1,Npar):
    #             if i!=j:
    #                 fig = plt.figure(Npar*i+j,figsize=(3.5, 2.8))
    #                 ax = fig.gca()

    #                 ax.grid(True)
    #                 ax.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
    #                 ax.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')


    #                 ax.set_xlim([np.min(statSmp[i,:]),np.max(statSmp[i,:])])
    #                 ax.set_ylim([np.min(statSmp[j,:]),np.max(statSmp[j,:])])

    #                 # ax.set_xlim([X_low[i],X_up[i]])
    #                 # ax.set_ylim([X_low[j],X_up[j]])

    #                 vmin = -220; vmax = np.nanmax(logLik[:,:,countdim])
    #                 print(np.min(logLik[:,:,countdim])); print(np.max(logLik[:,:,countdim]))
    #                 levels = np.linspace(vmin, vmax, 30)
    #                 cont=ax.contourf(p1[:,:,countdim],p2[:,:,countdim],logLik[:,:,countdim],cmap='Greens',vmax=vmax,vmin=vmin)
    #                 samples=ax.plot(Xsmp[i,:],Xsmp[j,:],'r.',markersize = msize)
    #                 ## or contour plot ontop of contourf plot:
    #                 #cont=ax.contour(p1[:,:,countdim],p2[:,:,countdim],logLik[:,:,countdim],colors='k',vmax=vmax,vmin=vmin,extend='min',levels=levels)
    #                 # ax.grid(True)
    #                 # ax.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
    #                 # ax.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
    #                 # ax.set_xlim([X_low[i],X_up[i]])
    #                 # ax.set_ylim([X_low[j],X_up[j]])
    #                 # ax.set_xlim([phiTrue[i]-0.5,phiTrue[i]+0.5])
    #                 # ax.set_ylim([phiTrue[j]-0.5,phiTrue[j]+0.5])
    #                 fig.colorbar(cont)
    #                 ax.set_xlabel(mylabel[i])
    #                 ax.set_ylabel(mylabel[j])
    #                 fig.savefig('/Users/sudhipv/documents/coupledode/codes/Inference/parallel_TMC/mobility/figs/Jloglik'+str(i)+str(j)+'.eps',bbox_inches='tight')
    #                 countdim = countdim+1
    #                 plt.close()

    # #TODO: add a section where the logliklihood function is computed without using any MAP estimates. For a 2-parameter case, we can get away with this.
    # if ((compgrd == 1) or (compgrd == 2)) and (Npar==2):
    #     logLikInt = np.zeros((Npar,nl))
    #     cur_x = np.zeros((2,nl))
    #     for i in range(0,Npar): #i represents the current parameter we want to integrate out.
    #         #upper and lower limits of x axis for integration:
    #         xmin = np.min(statSmp[i,:]) # X_low[i]
    #         xmax = np.max(statSmp[i,:]) # X_up[i]
    #         cur_x[i,:] = np.linspace(xmin,xmax,nl)
    #         for ii in range(0,nl): # go along every entry in the grid and integrate out the parameter.
    #             if i==0: #Integrate along the rows of the logliklihood function grid
    #                 cur_y = np.exp(logLik[:,ii])
    #                 cur_y = np.reshape(cur_y,(nl,))
    #                 logLikInt[i,ii] = trapezoid(y=cur_y,x=cur_x[i,:],dx=(cur_x[1]-cur_x[0]))
    #             elif i==1: #integrate along the columns of the logliklihood function grid
    #                 cur_y = np.exp(logLik[ii,:])
    #                 cur_y = np.reshape(cur_y,(nl,))
    #                 logLikInt[i,ii] = trapezoid(y=cur_y,x=cur_x[i,:],dx=(cur_x[1]-cur_x[0]))

    # ## Plot the integrated plots against the KDE estimate of the parameter posterior pdf

    # if ((compgrd == 1) or (compgrd ==2)) and (Npar >= 2):
    #     scaleFactor = 1
    #     for i in range(0,Npar):
    #         fig = plt.figure(1+i,figsize=(3.5, 2.8))
    #         ax = fig.gca()
    #         # Logliklihood function:
    #         plt1=ax.plot(cur_x[i,:],logLikInt[i,:],'g-',label=r'likelihood') #label=r'Log likelihood'
    #         myYTicks = []
    #         #myXlim = [np.min(statSmp[i,:]),np.max(statSmp[i,:])]
    #         ax.set_yticks(myYTicks)
    #         # myXlim = np.array([scaleFactor*np.min(statSmp[i,:]),scaleFactor*np.max(statSmp[i,:])])
    #         # ax.set_xlim(myXlim)

    #         myXlim = np.array([scaleFactor*np.min(statSmp[i,:]),scaleFactor*np.max(statSmp[i,:])])
    #         # myXlim = np.array([phiTrue[i]-0.2,phiTrue[i]+0.2])
    #         ax.set_xlim(myXlim)
    #         maxlik = np.max(logLikInt[i])
    #         minlik = np.min(logLikInt[i])
    #         myYLim = [0,maxlik*1.1]
    #         #  myYLim = [0,maxlik*1.1]
    #         ax.set_ylim(myYLim)
    #         ax.grid(True)
    #         ax.set_xlabel(mylabel[i])
    #         ax.set_ylabel(r'likelihood')
    #         #KDE plot ontop of the marginal plot
    #         ax2=ax.twinx()
    #         xlow, xup = np.min(statSmp[i,:]), np.max(statSmp[i,:])
    #         Xpdf = st.gaussian_kde(statSmp[i,:],bw_method = 0.3)  ## adjust kernel width
    #         Xgrd = np.linspace(xlow,xup,nl)
    #         plt2=ax2.plot(Xgrd,Xpdf(Xgrd),'b-',linewidth=0.5,label=r'KDE')
    #         #plot Histogram:
    #         ax3 = ax2.twinx()
    #         Nbins = int(np.sqrt(Nsmp))
    #         y,x,_ = ax3.hist(statSmp[i,:],alpha=0.1,bins=Nbins) #y and x return the bin locations and number of samples for each bin, respectively
    #         myYlim3 = [0,1.1*y.max()]
    #         ax3.set_ylim(myYlim3)
    #         ax3.set_yticks([])
    #         maxpdf = np.max(Xpdf(Xgrd))
    #         myYLim2 = [0,1.1*maxpdf]
    #         ax2.set_ylim(myYLim2)
    #         ax2.set_xlim(myXlim)
    #         ax2.set_ylabel(r'pdf')
    #         ax2.set_yticks(myYTicks)
    #         # Legend
    #         if i==0:
    #             plt3=ax.plot([phiTrue[i],phiTrue[i]],myYLim,'k--',label='True')
    #             plts = plt1+plt2+plt3
    #             labs = [l.get_label() for l in plts]
    #             ax.legend(plts, labs, loc='upper right',fontsize=10)
    #         else:
    #             plt3=ax.plot([phiTrue[i],phiTrue[i]],myYLim,'k--')
    #         fig.savefig('/Users/sudhipv/documents/coupledode/codes/Inference/parallel_TMC/mbility/figs/m_loglikInt'+str(i)+'.pdf',bbox_inches='tight')
    #         plt.close()



###################### ###################### ###################### ###################### ######################


    msize = 1.2
    for i in range(0,Npar):
        for j in range(i+1,Npar):
            plt.figure(Npar*i+j,figsize=(3.5, 2.8))
            plt.plot(Xsmp[i,:],Xsmp[j,:],'b.',markersize = msize)
            plt.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
            plt.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
            plt.xlabel(mylabel[i])
            plt.ylabel(mylabel[j])
            plt.xlim([np.min(statSmp[i,:]),np.max(statSmp[i,:])])
            plt.ylim([np.min(statSmp[j,:]),np.max(statSmp[j,:])])
            # plt.xlim([phiTrue[i]-0.1,phiTrue[i]+0.1])
            # plt.ylim([phiTrue[j]-0.1,phiTrue[j]+0.1])
            plt.grid(True)
            plt.savefig('./figs/Jsmpls_'+str(i+1)+str(j+1)+'.eps',bbox_inches='tight')
            plt.close()

    msize = 1.2
    for i in range(0,Npar):
        for j in range(i+1,Npar):
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
            plt.plot([X_low[i],X_up[i]],[phiTrue[j],phiTrue[j]],'r--')
            plt.plot([phiTrue[i],phiTrue[i]],[X_low[j],X_up[j]],'r--')
            plt.xlabel(mylabel[i])
            plt.ylabel(mylabel[j])
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            # plt.xlim([phiTrue[i]-0.1,phiTrue[i]+0.1])
            # plt.ylim([phiTrue[j]-0.1,phiTrue[j]+0.1])
            plt.grid(True)
            plt.savefig('./figs/jpdf_'+str(i+1)+str(j+1)+'.eps',bbox_inches='tight')
            plt.close()

    # kdeMCMC= st.gaussian_kde(statSmp,bw_method = 0.1)
    # SigMat = kdeMCMC.covariance
    # np.savetxt('SigMat.dat',SigMat)
