#!/usr/bin/env python
# coding: utf-8
#Author: Karssien Hero Huisman

from matplotlib import pyplot as plt
import numpy as np


import negf_HIA_git                    ### NEGF methods for Hubbard One Green's functions
import handy_functions_coulomb as hfc  ### some usefull functions are imported

        
        

def calc_bound_integrand_current(Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR,nround ):
    
    '''
    Input:
    nround = numerical order one wants to neglect
    Output:
    lower and upper bound [elower,eupper] outside which the integrand is to the numerical order given by nround.
    '''
    
    elist = [muL,muR]
    e_lower = min(elist) - 1
    
    
    
    integrand_lower = abs( np.round( negf_HIA_git.integrand_current_HIA(e_lower,
                                 Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR ),nround
                                   )
                         ) 
    
    
    
    
    while integrand_lower != 0:
        e_lower -= 0.1
        integrand_lower = abs( np.round( negf_HIA_git.integrand_current_HIA(e_lower,
                                 Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR ),nround
                                   )
                         ) 
        
        
        
        
    e_upper = max(elist) + 1
    
    integrand_upper = abs( np.round( negf_HIA_git.integrand_current_HIA(e_upper,
                                 Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR )
                                    ,nround 
                                   )
                         )
    
    
    while integrand_upper != 0:
        e_upper += 0.1
        integrand_upper = abs( 
                        np.round( 
                               negf_HIA_git.integrand_current_HIA(e_upper,
                                 Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR )
                                 ,nround 
                                )
                         )
    
    
    
    
    return e_lower,e_upper


def calc_I_trapz(npoints_int,
                V_list,ef,
              Hamiltonian0 ,
              GammaL,GammaR,
              U,n_tot_list,
              betaL,betaR):
    
    '''
    Input:
    npoints_int = number of points to integrate over
    - V_list = list of voltages (positive and negative )
    - ef = Fermi energy
    - Hamiltonian0 = hamiltonian for U = 0 
    - GammaL,GammaR: Gamma matrices left,rigth lead
    - U =  Coulomb interaction strength
    - n_list_total = list of electron densities 
    - betaL, betaR = beta =1/kBT of left, right lead
    Output:
    Ilist = list of currents for voltage V
    '''
       
    I_list = []
    
    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2,ef-V/2
        n_list = n_tot_list[i]
        
        #Cut off integrand that does not contribute significantly.
        elower,eupper = calc_bound_integrand_current(Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR,15 )

        energies = np.linspace(elower,eupper,npoints_int)

        
        
        Tbar_list = [ negf_HIA_git.integrand_current_HIA(energy,
                                 Hamiltonian0 ,
                                  GammaL,GammaR,
                                  U,n_list, 
                                  betaL,betaR,muL,muR ) for energy in energies]
        
#         plt.plot(energies,Tbar_list)
#         plt.show()

        Icur = np.trapz(Tbar_list,energies)
        I_list.append(Icur)




    return I_list
                     


        
def func_MR_list(y1list,y2list,xlist):
    '''Input
    y1list,y2list: lists that are a function of the parameter x in xlist
    Output
    plist = list with values: 'P = (y2-y1)/(y1 + y2)' 
    xprime_list = New x parameters. Values of x for which y1(x) + y2(x) =0 are removed (0/0 is numerically impossible)'''
    
    p_list = []
    xprime_list= []
    for i in range(len(xlist)):
        x = xlist[i]
        
        y1 = y1list[i]
        y2 = y2list[i]
        
        
        
        if x!=0:
            p_list.append(100*np.subtract(y1,y2)/(y2 + y1))
            
            xprime_list.append(x)
            
    
    return xprime_list,p_list



