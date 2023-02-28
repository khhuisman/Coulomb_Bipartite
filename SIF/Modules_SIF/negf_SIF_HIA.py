#!/usr/bin/env python
# coding: utf-8

### Code written by : Karssien Hero Huisman 13-12-2022


from matplotlib import pyplot as plt
import numpy as np
import handy_functions_coulomb as hfc

########################################################################################################## Fermi Functions
#################################################################################################

#beta
def func_beta(T):
    kB = 8.617343*(10**(-5)) # eV/Kelvin
    ''' Input:
    -Temperature in Kelvin
    Output:
    - beta in eV^-1
    '''
    if T > 0:
        return 1/(kB*T)
    if T ==0:
        return np.infty
    if T<0:
        print('Temperature cannot be negative')

# Fermi-Dirac function
def fermi_dirac(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The fermi-Dirac distribution for energy 
    '''
    if beta < 0:
        print('Error: Beta must be positive)')
    elif beta == np.infty:
        # return heavi-side function if T=0 
        fd = np.heaviside(-(energy-mui),1) 
        return fd
    else:
        deltae = energy-mui
        if np.sign(deltae) == -1:
            fd = 1/(np.exp(beta*(energy-mui) ) + 1 )
            return fd
        
        if np.sign(deltae) == 1:
            fd = np.exp(beta*(mui-energy) )/(np.exp(beta*(mui-energy) ) + 1 )
            return fd
        
        if np.sign(deltae) == 0:
            fd = 1/(np.exp(beta*(energy-mui) ) + 1 )
            return fd
    
    
    
# Derivative of Fermi-Dirac function
def fermi_prime_dirac(energy,mui,beta,mui_prime):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:

        fd = mui_prime*beta*(1/4)*(1/np.cosh(beta*(energy-mui)/2))**2 
        return fd
    
def fermi_pp_dirac(energy,mui,beta,mui_prime):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:
        
#         if energy != mui:
#             fd = mui_prime*(2*beta**2)*((1/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui))**4)
#             return fd
        
        if energy != mui:
            fd = mui_prime*(2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
            return fd
        if energy == mui:
        
            return 0
    
    
def de_fermi_dirac(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:

        fd = -1*beta*(1/4)*(1/np.cosh(beta*(energy-mui)/2))**2 
        return fd
    
def de2_fermi_dirac(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:
        
#         if energy != mui:
#             fd = mui_prime*(2*beta**2)*((1/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui))**4)
#             return fd
        
        if energy != mui:
            fd = (2*beta**2)*((np.sinh(0.5*beta*(energy-mui))/np.sinh(beta*(energy-mui)))**3)*(np.sinh(0.5*beta*(energy-mui)))
            return fd
        if energy == mui:
        
            return 0    
    
    
# Derivative of Fermi-Dirac function
def fermi_prime_dirac_heat(energy,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The "derivative of the fermi-Dirac w.r.t chemical potential" distribution for energy 
     taken from equation (12) "Thermoelectric Signatures of Coherent Transport in Single-Molecule Heterojunctions,2009"
    '''
    
    if beta < 0:
        print('Error: Beta must be positive)')
    if beta >0:
        fd = (beta/4)*(1/(np.cosh(beta*(energy-mui)/2)**2) )
        return fd



# Calculate inverse of matrix M:
def inverse(M):
    invers_M = np.array(np.linalg.inv(M),dtype = complex)
    return invers_M

#Advandced, Retarded Green's Function, with semi-infinite leads

def Sigma_imag(energy,epsilon0,tlead):
    
    D1 =  (2*tlead)**2 -(energy - epsilon0)**2 
    
    if D1 > 0:
        return np.sqrt(D1)
    
    if D1 <= 0:
        return 0
    
    
def Derivative_Sigma_imag(energy,epsilon0,tlead):
    
    D1 =  (2*tlead)**2 -(energy - epsilon0)**2 
    
    if D1 > 0:
        return -(energy - epsilon0)/np.sqrt(D1)
    
    if D1 <= 0:
        return 0
    
    
def Sigma_real(energy,epsilon0,tlead):
    
    Dreal =   ((energy - epsilon0)/2)**2 - (tlead)**2 
    

    
    if Dreal < 0:
        return (energy - epsilon0)/2
    
    if Dreal >= 0:
        asign = np.sign((energy - epsilon0)/2)
        if asign > 0:
            return (energy - epsilon0)/2 - np.sqrt(Dreal)
        
        if asign < 0:
            return (energy - epsilon0)/2 + np.sqrt(Dreal)
    

    
def Derivative_Sigma_real(energy,epsilon0,tlead):
    
    Dreal =   ((energy - epsilon0)/2)**2 - (tlead)**2 
    
    if Dreal < 0:
        return 1/2
    
    if Dreal >= 0:
        asign = np.sign((energy - epsilon0)/2)
        if asign > 0:
            return 1/2 - ((energy - epsilon0)/2)/np.sqrt(Dreal)
        
        if asign < 0:
            return 1/2 + ((energy - epsilon0)/2)/np.sqrt(Dreal)
    
    
def Sigma_Retarded(energy,epsilon0,tlead,tcoup):
    Lambda = Sigma_real(energy,epsilon0,tlead) 
#     Lambda = 0


    Gamma  = Sigma_imag(energy,epsilon0,tlead)


    
    Sigma = ((tcoup/tlead)**2)*(Lambda - (1j/2)*Gamma)
    return Sigma  

def Mat_Sigma_Retarded(energy,epsilon0,tlead,tcoup,mat):
    return np.array( np.multiply(Sigma_Retarded(energy,epsilon0,tlead,tcoup),mat),dtype = complex)





def SigmaPM(energy,epsilon0,tlead,tcoup,pz):
    '''
    Input:
    epsilon0  = Onsite energy of left,right lead respectively
    tlead        = Hopping parameter of left,right lead respectively
    Output:
    Self energy for the up,down electrons
    '''
    
    #Retarded Self Energy
    Sigma_up  = Sigma_Retarded(energy,epsilon0 + pz*2*tlead ,tlead,tcoup)
    Sigma_dwn = Sigma_Retarded(energy,epsilon0 - pz*2*tlead,tlead,tcoup)
    
    
    
    return Sigma_up,Sigma_dwn



def plot_selfenergies(energies,epsilon0,tlead,tcoup,pz):
    '''
    Input:
    energies = list of energies
    epsilon0 = onsite energy of semi-infite lead
    tlead = hopping paramters semi-infinite lead (NN)
    tcoup = coupling paramter between molecule and lead
    pz = magnetization of th lead
    Output:
    plot of real and imaginary part of the retarded self energy for the different spin-species
    '''
    SigmaL_list = [ SigmaPM(energy,epsilon0,tlead,tcoup,pz)  for energy in energies]
    Gamma_ulist = [ (-2)*SigmaL_list[i][0].imag for i in range(len(energies))]
    Gamma_dlist = [ (-2)*SigmaL_list[i][1].imag for i in range(len(energies))]


    Lambdalist_up = [  SigmaL_list[i][0].real for i in range(len(energies))]
    Lambdalist_down = [  SigmaL_list[i][1].real for i in range(len(energies))]



    plt.plot(energies,Gamma_ulist,label = '$\Gamma_u$')
    plt.plot(energies,Gamma_dlist,label = '$\Gamma_d$')
    plt.xlabel('Energy')
    plt.ylabel('$\Gamma$')
    plt.legend()
    plt.show()


    plt.plot(energies,Lambdalist_up,  label = '$\Lambda_u$')
    plt.plot(energies,Lambdalist_down,label = '$\Lambda_d$')
    plt.xlabel('Energy')
    plt.ylabel('$\Lambda$')
    plt.legend()
    plt.show()



def Sigma(energy,
                epsilon0L,tleadL,tcoupL,matL,
                epsilon0R,tleadR,tcoupR,matR,pz):
    '''
    Input:
    epsilon0L.epsilon0LR = Onsite energy of left,right lead respectively
    tleadL,tleadR        = Hopping parameter of left,right lead respectively
    matL,matR = Matrices which tells how molecule and lead are coupled
    '''
    
    #Retarded Self Energy
    SigmaLu = Mat_Sigma_Retarded(energy,epsilon0L + pz*2*tleadL ,tleadL,tcoupL,matL)
    SigmaLd = Mat_Sigma_Retarded(energy,epsilon0L - pz*2*tleadL,tleadL,tcoupL,matL)
    
    SigmaLspin = np.add( np.kron(SigmaLu,np.diag([1,0])),
                    np.kron(SigmaLd,np.diag([0,1]))
                      )
                                                           
    SigmaR = Mat_Sigma_Retarded(energy,epsilon0R,tleadR,tcoupR,matR)
    SigmaRs = np.kron(SigmaR,np.identity(2))
    
    Sigma_ret_spin = np.add(SigmaLspin,SigmaRs)

    
    #ImaginaryEnergies
    GammaL = -2*SigmaLspin.imag
    GammaR = -2*SigmaRs.imag
    
    return Sigma_ret_spin,GammaL, GammaR



########################################################################################################## Retarded, Advanced Green's function
#################################################################################################



def GRA_HIA(energy, 
        H,
        SigmaP,
         U,n_list):
    
    '''
    Input
    - energy = energy
    - H      = Hamiltonian of scattering region for U = 0 (no leads)
    - SigmaP = Retarded Self energy
    - U      = Coulomb interaction strength
    - n_list = list of electron densities
    
    Output:
    - Retarded,Advanced Green's Function in WBL and the Hubbard I approximation
    '''
    
    def GP_HIA(energy, 
        H,
        SigmaR,
         U,n_list):
    
        # Hubbard 1 approximation Green's function:
        # GR =
        # [ 
        #                         (E - ES - Umat) (E - H0 - (i/2)Gamma)
        #                    - nU (t + v + (i/2)Gamma)
        #                   ]^-1 X [ E-ES - Umat(1-n)]
        #
        # ES = diagonal matrix with onsite energies
        # Umat = diagonal matrix with U on the diagonal
        # H0 = Full Hamiltonian
        # SigmaR = retarded self energy
        # t,v = hopping matrix and spin-orbit coupling hopping matrix respectively.
        # nU = diagonal matrix: U ni\bars \delta_ij \delta_{s,sp}
        
        n_swapped = hfc.pairwise_swap(n_list)               # = [n1d,n1u,n2d,n2u,....] densities are swapped per site.
        n_list_cor =  n_swapped

        h_dim = np.shape(H)[1]

        En = np.array((energy+U/2)*np.identity(h_dim),dtype = complex)

        ES = np.diag(H,0)
        Umat = U*np.identity(h_dim)
        nmat = U*np.diag(n_list_cor) #Un

        EH = np.subtract(En,H)
        Gamma = SigmaP




        En1 = np.subtract(En, np.add(ES,Umat))   # (E - ES - Umat)
        En2 = np.subtract(En, np.add( H, Gamma)) # (E - H0 - Sigma)
        X   = np.add(np.subtract(H, ES), Gamma)  # (t + v + Sigma)
        En3 = np.dot(nmat,X) # Un(t + v + Sigma)
        
        
        Noemer = np.subtract(np.dot(En1,En2),En3) # [ (E - ES - U)(E - H0 - Sigma) - Un(t + v + Sigma) ]
        G_inv = inverse(Noemer)
        teller = np.add(np.subtract(En,np.add(ES,Umat)), nmat) # E - ES - U + Un


        #Retarded, advanced Green's function
        GR = np.dot(G_inv,teller) #G+
        
        
        return GR
    
    
    
    G_R = GP_HIA(energy, 
        H,
        SigmaP,
         U,n_list)
    
    G_A = np.conjugate(np.transpose(G_R))

    
  
    return [G_R,G_A]




    
########### Sigma Lesser function ###########

def SigmaLesser(energy,
                GammaL,GammaR,
                muL , muR,
                betaL,betaR):
    
    
    shape = GammaL.shape
    SigmaLess = np.zeros(shape)
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    # Equation (2.123) from Seldenthuys thesis:
        #Sigma<_alpha = 1j Gamma_alpha*f_alpha 
        #Sigma<       = Sigma<_L + Sigma<_R
        #Sigma<       = 1j( GammaL*fL+ GammaR*fR)
    
    SigmaLess = np.multiply(1j,
                                np.add(
                                        np.multiply(GammaL,fL), 
                                        np.multiply(GammaR,fR)
                                      )
                           )
    
    return SigmaLess





def GLesser(energy,H,U,n_list,
                epsilon0L,tleadL,tcoupL,matL,
                epsilon0R,tleadR,tcoupR,matR,
                pz,
                muL, muR,
                betaL,betaR):
    
    
    
    
    Sigma_ret,GammaLs,GammaRs = Sigma(energy,
                                    epsilon0L,tleadL,tcoupL,matL,
                                    epsilon0R,tleadR,tcoupR,matR,
                                     pz)
    
   
    
    
    GR,GA =  GRA_HIA(energy, 
                    H,
                    Sigma_ret,
                     U,n_list)
    
    
    SigmaLess = SigmaLesser(energy,GammaLs,GammaRs,
                            muL, muR ,
                            betaL,betaR)
    
    
    
    
    # G< = GR.Sigma<.GA = GR.[GammaL*fL + GammaR*fR].GA
    Gless = np.dot(GR,np.dot(SigmaLess,GA) )
    
    return Gless






def ndensity_listi(energy,
                 Hamiltonian0,U,n_list,
                epsilon0L,tleadL,tcoupL,matL,
                epsilon0R,tleadR,tcoupR,matR,
                   pz,
                muL, muR,
                betaL,betaR):
    
    
    '''
    Input:
    - energy         = energy
    - Hamiltonian0   = Hamiltonian of scattering region for U = 0
    - GammaL,GammR   = Gamma Matrices of left,right lead.
    - muL,muR        = chemical potential of the left,right lead.
    - betaL,betaR    = beta = (kBT)^-1 of left,right lead.
    Ouput
    - List of electron densities on site i with given a certain energy.
    '''
    
    
    Glessim = np.multiply(-1j/(2*np.pi),GLesser(energy,Hamiltonian0,U,n_list,
                                        epsilon0L,tleadL,tcoupL,matL,
                                        epsilon0R,tleadR,tcoupR,matR,
                                        pz,
                                        muL, muR,
                                        betaL,betaR) 
                         ).real
    
    n_ilist = np.diag(Glessim,0)
    
    
    return n_ilist







# ########### Local Density of states ###########



def LDOS(energy,H,U,n_list,
                epsilon0L,tleadL,tcoupL,matL,
                epsilon0R,tleadR,tcoupR,matR,
                    pz,
                muL, muR,
                betaL,betaR):
    
    Sigma_ret, GammaL,GammaR = Sigma(energy,
                                            epsilon0L,tleadL,tcoupL,matL,
                                            epsilon0R,tleadR,tcoupR,matR,
                                                                 pz)
    
    GR,GA =  GRA_HIA(energy, 
                    H,
                    Sigma_ret,
                     U,n_list)
    
    A = 1j*np.subtract(GR,GA)
    
    LDOS = (np.matrix.trace(A)/(2*np.pi)).real
    
    return LDOS


def plot_dos(energies,H,U,n_list,
                epsilon0L,tleadL,tcoupL,matL,
                epsilon0R,tleadR,tcoupR,matR,
                    pz,
                muL, muR,
                betaL,betaR):
    '''
    Input:
    energies = list of energies
    epsilon0 = onsite energy of semi-infite lead
    tlead = hopping paramters semi-infinite lead (NN)
    tcoup = coupling paramter between molecule and lead
    pz = magnetization of th lead
    Output:
    plot of real and imaginary part of the retarded self energy for the different spin-species
    '''
    DOSlist = [ LDOS(energy,H,U,n_list,
                epsilon0L,tleadL,tcoupL,matL,
                epsilon0R,tleadR,tcoupR,matR,
                    pz,
                muL, muR,
                betaL,betaR)  for energy in energies]
    
    
   



    plt.plot(energies,DOSlist)
    plt.xlabel('Energy')
    plt.ylabel('DOS')
#     plt.legend()
    plt.show()





###################################################################
############ Integrand for calculating 2 terminal current  ########
###################################################################

####### Transmission for strictly 2-Terminal systems  #######

#Transmission left to right
# Only valid for 2terminal junctions
def TLR_semi_inf(energy, 
        H,U,n_list,
        epsilon0L,tleadL,tcoupL,matL,
        epsilon0R,tleadR,tcoupR,matR,
                 pz):
    
    Sigma_ret, GammaL,GammaR = Sigma(energy,
                                        epsilon0L,tleadL,tcoupL,matL,
                                        epsilon0R,tleadR,tcoupR,matR,
                                     pz)
    
    
    
    GR,GA =  GRA_HIA(energy, 
                    H,
                    Sigma_ret,
                     U,n_list)
    
    T = np.dot(np.dot(np.dot(GammaL,GA),GammaR),GR)
    T_LR = np.matrix.trace(T).real
    
    
    return T_LR


def integrand_current_HIA(energy, H,U,n_list,
                                    epsilon0L,tleadL,tcoupL,matL,
                                    epsilon0R,tleadR,tcoupR,matR,
                      pz,
                          betaL,betaR,muL,muR):
    
    '''
    Input:
    System paramters (see ChiralChainModel.py for description of all)
    - energy of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead
    
    
        
    
    Output:
    - Current calculated with Landauer-Buttiker Formula '''
        
    
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    integrand = TLR_semi_inf(energy, 
                            H,U,n_list,
                            epsilon0L,tleadL,tcoupL,matL,
                            epsilon0R,tleadR,tcoupR,matR,
                             pz)*(fL-fR)
    
    return integrand







    
def plot_selfenergies_voltage_bothmagnetizations(energies,ef,V,tlead,tcoup,pz):
    '''
    Input:
    energies = list of energies
    epsilon0 = onsite energy of semi-infite lead
    tlead = hopping paramters semi-infinite lead (NN)
    tcoup = coupling paramter between molecule and lead
    pz = magnetization of th lead
    Output:
    
    plot up,down component of Gamma for +m and -m as function of bias voltage V.
    
    '''
    
    ###Left Lead
    SigmaL_list_max = [ SigmaPM(energy,ef + V/2,tlead,tcoup,pz)  for energy in energies]
    SigmaL_list_min = [ SigmaPM(energy,ef + V/2,tlead,tcoup,-pz)  for energy in energies]
    
    GammaL_ulist_max = [ (-2)*SigmaL_list_max[i][0].imag for i in range(len(energies))]
    GammaL_dlist_max = [ (-2)*SigmaL_list_max[i][1].imag for i in range(len(energies))]
    
    GammaL_ulist_max = [ (-2)*SigmaL_list_min[i][0].imag for i in range(len(energies))]
    GammaL_dlist_max = [ (-2)*SigmaL_list_min[i][1].imag for i in range(len(energies))]
    

    plt.plot(energies,GammaL_ulist_max,label = '$\Gamma_u(m)$')
    plt.plot(energies,GammaL_dlist_max,label = '$\Gamma_d(m)$')
    plt.plot(energies,GammaL_ulist_max,label = '$\Gamma_u(-m)$')
    plt.plot(energies,GammaL_dlist_max,label = '$\Gamma_d(-m)$')
    plt.xlabel('Energy')
    plt.ylabel('$\Gamma$')
    plt.legend()
    plt.show()
    
    
def plot_selfenergies_voltage_one_magnetization(energies,ef,V,tlead,tcoup,pz):
    '''
    Input:
    energies = list of energies
    epsilon0 = onsite energy of semi-infite lead
    tlead = hopping paramters semi-infinite lead (NN)
    tcoup = coupling paramter between molecule and lead
    pz = magnetization of th lead
    Output:
    
    plot up,down component of Gamma for +m and -m as function of bias voltage V.
    
    '''
    
    ###Left Lead
    SigmaL_list_max = [ SigmaPM(energy,ef + V/2,tlead,tcoup,pz)  for energy in energies]
    
    GammaL_ulist_max = [ (-2)*SigmaL_list_max[i][0].imag for i in range(len(energies))]
    GammaL_dlist_max = [ (-2)*SigmaL_list_max[i][1].imag for i in range(len(energies))]
    
    
    
    plt.title('Vbias = {}'.format(V))
    plt.plot(energies,GammaL_ulist_max,label = '$\Gamma_u(m)$')
    plt.plot(energies,GammaL_dlist_max,label = '$\Gamma_d(m)$')
    plt.xlabel('Energy')
    plt.ylabel('$\Gamma$')
    plt.legend()
    plt.show()



