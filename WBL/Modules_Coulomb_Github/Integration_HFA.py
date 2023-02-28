
#!/usr/bin/env python
# coding: utf-8

# Author: Karssien Hero Huisman
# Module: 
# 1) Self-consistently calculates electron densities in HFA approximation.
# 2) Uses "Linear Mixing " scheme 

import numpy as np
from matplotlib import pyplot as plt


import handy_functions_coulomb as hfc ###### some usefull functions 
import negf_git                       ###### import NEGF method

        



################################################################################
######## Calculate Electron Density  
################################################################################

def check_integrand_zero(energies,
                         HamiltonianU,
                         GammaL,GammaR,
                        muL, muR,
                        betaL,betaR,tol_nintegrand):
    
    '''
    Input:
    - energies = list of energies
    - U = Coulomb interaction strength
    - HamiltonianU : Hamiltonian of scattering region for U finite
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    - tol_nintegrand = cutoff value on integrand
    Output:
    - check_zerob  = list of round values for lower,upperbound
    - boundbool = boolean. True if integrand is numerically neglible. False if it is not.
    '''
    
    emin,emax = min(energies),max(energies)

    nlist_min = negf_git.ndensity_listi(emin,
                                             HamiltonianU,
                                             GammaL,GammaR,
                                            muL, muR,
                                            betaL,betaR)



    nlist_max = negf_git.ndensity_listi(emax,
                                             HamiltonianU,
                                             GammaL,GammaR,
                                            muL, muR,
                                            betaL,betaR)





    nlist_minmax = hfc.jointwolist(nlist_min,nlist_max)
    

    check_zerob, boundbool = hfc.check_listzero(nlist_minmax,tol_nintegrand)
    
    return check_zerob, boundbool

def calc_electron_density_trapz(energies,
                         HamiltonianU,
                         GammaL,GammaR,
                        muL, muR,
                        betaL,betaR,tol_nintegrand):


    '''
    Input:
    - energies = list of energies
    - listi = list of labels corresponding to orbitals with coulomb interactions
    - U = Coulomb interaction strength
    - Hamiltonian : Hamiltonian of scattering region with Coulomb interactions
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - list of electron densities for site 1,2,3,4...
    - list of onsite hoppings between the different spin species. 
    '''


    ##### 
    nE00_list = []
    
    
    shape = HamiltonianU.shape
    
    ###check if Glesser function is negligable
    check_zerob, boundbool = check_integrand_zero(energies,
                         HamiltonianU,
                         GammaL,GammaR,
                        muL, muR,
                        betaL,betaR,tol_nintegrand)
    
    if boundbool == True:
        # This way one calculate GR only one time per energy instead of i times per energy. (i times less matrix inversions)
        for energy in energies:
            n00_list_en = negf_git.ndensity_listi(energy,
                                             HamiltonianU,
                                             GammaL,GammaR,
                                            muL, muR,
                                            betaL,betaR)



            nE00_list.append(n00_list_en)

        #integrate every electron density over all energies.    
        n00list_i = [ np.trapz(np.array(nE00_list)[:,i],energies) for i in range(shape[0])]


        return n00list_i
    
    if boundbool == False:
        
        print('Choose different upper/lower bound for integration, integrand of electron density larger than {}'.format(
                tol_nintegrand))
        
        print(check_zerob)


################################################################################
######## Self - Consistent Calculation
################################################################################


#### self consistent loop given a voltage
def iteration_calculate_mixing2(n00_list,
                        max_iteration ,energies,
                        U,
                        Hamiltonian0,
                        GammaL,GammaR, 
                        muL,muR,betaL, betaR,tol,tol_nintegrand,alpha,plot_bool,trackbool):
    
    '''Input
    - n0_list : list of electron densities 
    - max_iteration : maximum number of iterations
    - energies : list of energies (to integrate Glesser function over)
    - U = Coulomb interaction strength
    - Hamiltonian0 : Hamiltonian of scattering region for U=0
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    Ouput:
    - list of electron densities for every iteration
    - alpha = number between [0,1) quantifying "linear mixing". 
    - plot_bool = boolean. True -> <nis>,<c^+is cisbar> are plotted for every iteration.
    - trackbool = boolean. True -> <nis>,<c^+is cisbar> ared saved and plotted after convergence or maximum iteration has been reached.
    '''
    
    zero_bool = False
    zero_bool00 =  False
    
    if plot_bool == True:
        trackbool = False
    
    #electron densitylist
    nk_iterations_list= []
    nk_iterations_list2 = []
    list_i = hfc.func_list_i(Hamiltonian0)
    k_list = [i for i in range(max_iteration)]
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    
    for k in k_list:
        # Hamiltonian with Hubbard terms
        HamiltonianU = negf_git.Hamiltonian_HF(n00_list,U,Hamiltonian0)

        # Calculate relevant electron densities:
        n00list_new =  calc_electron_density_trapz(energies,
                                                                           HamiltonianU,GammaL,GammaR,
                                                                           muL, muR, betaL,betaR,tol_nintegrand)
        
        

        if k > 0 and plot_bool == True:
            nk_iterations_list.append(n00list_new)
            if len(nk_iterations_list) > 0:
                plt.title('nis : Real')
                plt.plot(np.array(nk_iterations_list).real)
                plt.show()
            

        #check if values have converged
        check_zero00, zero_bool00 = hfc.check_difference2(n00_list,n00list_new,tol)
        
        
      
        print(np.round(check_zero00,acc))
        
        if trackbool == True:
            nk_iterations_list.append(n00list_new)
        

        
        if zero_bool00 == True:
            
            
            if trackbool == True or plot_bool == True:
                plt.title('nis : Real')
                plt.plot(np.array(nk_iterations_list).real)
                plt.show()
                
            
            zero_bool = True
            
            
            nk_iterations_list.append(n00list_new)
            break
            return nk_iterations_list,zero_bool
        
        
            
            
        
        if k == max_iteration-1:
            
            if trackbool == True or plot_bool == True:
                plt.title('nis : Real')
                plt.plot(np.array(nk_iterations_list).real)
                plt.show()
                
          

   
            nk_iterations_list.append(n00list_new)
            return nk_iterations_list,zero_bool

        #Re-assign electrond densities for next loop
        if zero_bool00 == False:
            
            n00_list = np.add( np.multiply(1-alpha,n00list_new), np.multiply(alpha,n00_list))

        
    return nk_iterations_list,zero_bool





######## Self - Consistent Calculation: loop over all voltages
def self_consistent_trapz_mixing_in(V_list,Vmax,
                                  n00_list_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energiesreal,tol_nintegrand,alpha,plot_bool,trackbool):
    
    '''Input
    - V_list = list of voltages
    - Vmax = maximum voltage
    - max_iteration : maximum number of iterations
    - n00_list_guess = initial guesses for electron densities.
    - ef = fermi energy 
    - U = Coulomb interaction strength
    - Hamiltonian0 : Hamiltonian of scattering region for U=0
    - GammaL,GammaR: Gamma matrices of left,right lead
    - muL,muR = chemical potential of left,right lead
    - betaL, betaR = beta =1/kBT of left, right lead
    - energies : list of energies (to integrate Glesser function over)
    Ouput:
    - list of self-consistently calculated electron densities for every voltage
    '''
    
    n_list = []
    convglist = []
    dim = Hamiltonian0.shape[0]
  
    
    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2, ef -V/2
        
        
        print ('--- V = {} ---'.format(V))
        if i ==0:
            n00_list_init = n00_list_guess
            

              

        if i !=0:
                
            n00_list_init = n_list[i-1] #initial guess.
            
        
        
        nlist_k,zero_bool = iteration_calculate_mixing2(n00_list_init,
                                                max_iteration ,energiesreal,
                                                  U,
                                                Hamiltonian0,
                                                GammaL,GammaR, 
                                                muL,muR,betaL, betaR,tol,tol_nintegrand,alpha,plot_bool,trackbool)

        
        n_list.append(nlist_k[-1])
        convglist.append(zero_bool)
        
        
    return n_list,convglist

        
######## Self - Consistent Calculation: loop over positive,negative
def self_consistent_trapz_PN(V_list_pos_bias,Vmax,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,
                                energiesreal,tol_nintegrand,alpha,plot_bool,trackbool):
    
    dim = Hamiltonian0.shape[0]
    pos_bool = hfc.check_list_smaller(np.sign(V_list_pos_bias),1)
    V_list_neg_bias = -1*V_list_pos_bias
    
    
    #V_list_pos_bias must be positive bias voltage.
    if pos_bool == True:
        
        #Intial guess for V = 0
        n00_V0_guess = hfc.list_halves(Hamiltonian0)
        
        
       
        
        
        #Start in equilibrium V = 0
        nV0_list,convglistV0 = self_consistent_trapz_mixing_in([0],0,
                                  n00_V0_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energiesreal,tol_nintegrand,alpha,plot_bool,trackbool)
        
        
        #Sweep for positive and negative bias voltages seperately.
        n00_V_guess = nV0_list[0]
        
        ##positive voltages
        n_list_VP,convglist_VP = self_consistent_trapz_mixing_in(V_list_pos_bias,Vmax,
                                  n00_V_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energiesreal,tol_nintegrand,alpha,plot_bool,trackbool)
         ##negative voltages
        n_list_VMprime,convglist_VMprime = self_consistent_trapz_mixing_in(V_list_neg_bias,Vmax,
                                  n00_V_guess,
                                  max_iteration,
                                ef,
                                U,
                                Hamiltonian0,
                                GammaL,GammaR, 
                                betaL, betaR,tol,energiesreal,tol_nintegrand,alpha,plot_bool,trackbool)
        
        n_list_VM = [n_list_VMprime[-1-i] for i in range(len(V_list_neg_bias))]
        convglist_VM = [ convglist_VMprime[-1-i] for i in range(len(V_list_neg_bias)) ] 

        #Join all densities,booleans and voltages into one list.
        n_list_total = hfc.jointwolist(hfc.jointwolist(n_list_VM,nV0_list),n_list_VP)
        convglist_total = hfc.jointwolist(hfc.jointwolist(convglist_VM,convglistV0),convglist_VP)
        return n_list_total,convglist_total
        

    if pos_bool == False:
        print(pos_bool)
        print('Bias voltages must be larger than zero')
        