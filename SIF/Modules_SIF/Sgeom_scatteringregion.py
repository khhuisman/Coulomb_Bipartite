#!/usr/bin/env python
# coding: utf-8

# In[15]:


import kwant
import kwant.qsymm
import matplotlib  
import numpy as np

# For matrix support
import tinyarray

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
a=1



def make_system_U0(t,tsom,Lm ,Wg):
    
    '''
    Input:
    - t = hopping parameter
    - tsom = spin - orbit coupling paramter
    - Lm,Wg = length, width of the S geometry
    Output:
    - kwant system of isolated,noninteracting S geometry
    
    '''
    
    
    #Define hoppings in z,x direction
    # The z - direction lies along lead direction
    # The x - direction is defined in-plane orthogonal to the z-direction
    # The y - direction points out-of-plane
    # Thus the lattice has coordinate label (z,x,y) (instead of the usual (x,y,z))
    
    # Create Lattice
    
    a   = 1
    lat =  kwant.lattice.cubic(a,norbs = 2) # 2 d.o.f. per site => norbs = 2
    syst = kwant.Builder()
    
    U = 0        #onsite energy
          #molecule length
    
     ### DEFINE LATTICE HAMILTONIAN ###
        
    
    d = Wg
    for i in range( -Lm , Lm + 1  ):
        for j in range(-d, d):
            
            
           
            # Sgeom is created:     
            if 0 <= i <= Lm and 0 <= j  <= Wg:
                syst[lat(i, j,0)] =  U*sigma_0
            if -Lm+1 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst[lat(i, j,0)] = U*sigma_0
              
           
    # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0,0), lat, lat)] =  -t*sigma_0 - 1j*tsom * sigma_x
    #hopping in x direction
    syst[kwant.builder.HoppingKind((0,1,0), lat, lat)] =  -t*sigma_0 + 1j* tsom* sigma_z
    
    return syst





def make_gammamatrices_wbl(gamma,pz,pz_R,Lm ,Wg):
    
    
    '''
    Input:
    - gamma  = coupling to leads
    - pz     = magnetic polarizaiton of left lead
    - pz_R   = magnetic polarization of right lead.
    - Lm,Wg  = length, width of the S geometry
    Output:
    - syst_left, syst_right: kwant systems for \Gamma matrices of left and right lead respectively. Used for the Wide band limit. Systems are constructed such that only the two leftmost and two rightmost sites are connected to these leads.
    '''
    
    #Define hoppings in z,x direction
    # The z - direction lies along lead direction
    # The x - direction is defined in-plane orthogonal to the z-direction
    # The y - direction points out-of-plane
    # Thus the lattice has coordinate label (z,x,y) (instead of the usual (x,y,z))
    
    # Create Lattice
    
    a   = 1
    lat =  kwant.lattice.cubic(a,norbs = 2) # 2 d.o.f. per site => norbs = 2
    syst_right = kwant.Builder()
    syst_left = kwant.Builder()


    
    
    ### DEFINE LATTICE HAMILTONIAN ###
    zeros =  0*sigma_0   
    
    d = Wg
    for i in range( -Lm , Lm + 1  ):
        for j in range(-d, d):
            
            
           
            # Sgeom is created:     
            if 0 <= i <= Lm-1 and 0 <= j  <= Wg:
                syst_right[lat(i, j,0)] = zeros
            if -Lm+2 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst_right[lat(i, j,0)] = zeros
                
           # Sgeom is created:     
            if 0 <= i <= Lm-1 and 0 <= j  <= Wg:
                syst_left[lat(i, j,0)] =  zeros
            if -Lm+2 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst_left[lat(i, j,0)] = zeros

            #left leadå
            if 0 <= j  <= Wg:
                if i == Lm:
                    syst_right[lat(i, j,0)] =  gamma*sigma_0 + gamma*pz_R*sigma_z
                    syst_left[lat(i, j,0)] =  zeros
            
            #right lead
            if -Wg+1 <= j  <= 0:
                if i == -Lm + 1:
                    syst_right[lat(i, j,0)] =  zeros
                    syst_left[lat(i, j,0)] =  gamma*sigma_0 + gamma*pz*sigma_z

    
    return syst_left, syst_right


def gamma_matrices(gamma,pz,Lm,Wg,pz_R=0):
    
    '''
    Input:
    - gamma  = coupling to leads
    - pz     = magnetic polarizaiton of left lead
    - pz_R   = magnetic polarization of right lead.
    - Lm,Wg  = length, width of the S geometry
    Output:
    - GammaLP,GammaLM = Gamma matrices of the Left lead for Postive and negative magnetization
    - GammaRP,GammaRM = Gamma matrices of the Right lead for Postive and negative magnetization
    
    '''
    
    #kwant systems: negative/positive magnetization.
    syst_leftP, syst_rightP = make_gammamatrices_wbl(gamma,abs(pz),abs(pz_R),Lm ,Wg)
    syst_leftM, syst_rightM = make_gammamatrices_wbl(gamma,-abs(pz),-abs(pz_R),Lm ,Wg)
   
    
    #GammaMatrices negative/positive magnetization.
    GammaLP = kwant.qsymm.builder_to_model(syst_leftP)[1]
    GammaLM = kwant.qsymm.builder_to_model(syst_leftM)[1]
    
    GammaRP = kwant.qsymm.builder_to_model(syst_rightP)[1]
    GammaRM = kwant.qsymm.builder_to_model(syst_rightM)[1]
    
    return GammaLP,GammaLM,GammaRP,GammaRM 



def hamiltonian_multiplesites_coupled(L,Wg, t,tsom,gamma,pz,plotbool=False):
    

    '''
    Input:
    L,Wg = geometric paramters of S shape
    t = hopping paramters
    tsom = spin-orbit coupling paramter
    gamma = coupling strenght to left,right lead.
    pz = magnetic polarization of the left lead. Note that |pz| <= 1 
    
    Ouput:
    Hamiltonian0    = Isolated Hamiltonian without interactions
    GammaR          = 2 times imaginary part of advanced self energy for right lead (Wide-band limit)
    GammaLP,GammaLM = 2 times imaginary part of advanced self energy for left lead (Wide-band limit) for positive,negative magnetization respectively
    '''
    
    hels3D =  make_system_U0(t,tsom,L ,Wg) 
    
    
    if plotbool == True:
        kwant.plot(hels3D);
    
    Hamiltonian0 = kwant.qsymm.builder_to_model(hels3D)[1]
    

    GammaLP,GammaLM,GammaR,GammaR = gamma_matrices(gamma,pz,Lm,Wg,0)

    return Hamiltonian0,GammaR,GammaLP,GammaLM



def hamiltonian_multiplesites_coupled_semi_inf(L,Wg, t,tsom,plotbool=False):
    
    '''
    Input:
    L,Wg = geometric paramters of S shape
    t = hopping paramters
    tsom = spin-orbit coupling paramter
    
    Ouput:
    Hamiltonian0    = Isolated Hamiltonian without interactions
    gammaR,gammaL   = matrix where all "off- diagonal" elements are zero. 
                     Diagonal elements are either:
                     - zero indicating that correspdong site is not coupled to the lead 
                     - or one (indicating that the sites is coupled to the lead).
                     
                     Matrix corresponds to: V d^+ c + V^+ c^+ d  
                     
                     Spin degree of freedom is not included in this module.
    '''
    
    
    hels3D =  make_system_U0(t,tsom,L ,Wg) 
    
#     if plotbool == True:
#         kwant.plot(hels3D);
    
    Hamiltonian0 = kwant.qsymm.builder_to_model(hels3D)[1]
    
    dim_gamma = int(Hamiltonian0.shape[0]/2)
   
    gammaL = np.zeros((dim_gamma,dim_gamma))
    gammaR = np.zeros((dim_gamma,dim_gamma))
    

   
    for j in range(0,Wg):
        gammaL[j,j] = 1
        gammaR[-1-j,-1-j] = 1


    return Hamiltonian0,gammaL,gammaR






########################################################
############## PHS matrices check ######################
########################################################

def make_sign_matrix(Lm ,Wg):
    a   = 1
    lat =  kwant.lattice.cubic(a,norbs = 2) # 2 d.o.f. per site => norbs = 2
    syst = kwant.Builder()        
    
    d = Wg
    for i in range( -Lm , Lm + 1  ):
        for j in range(-d, d):
            
            
           
            # Sgeom is created:     
            if 0 <= i <= Lm and 0 <= j  <= Wg:
                syst[lat(i, j,0)] =  ((-1)**(i+j))*sigma_0
            if -Lm+1 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst[lat(i, j,0)] = ((-1)**(i+j))*sigma_0
              
           
  
    
    return syst


def hamiltonian_signmatrix(Lm,Wg):
    '''
    Input:
    - Lm , Wg = geometric paramteters of of the S geometry
    Output:
    - Diagonal matrix U, which has +1 on sublattice A and -1 on sublattice B. (Matrix can used to verify -U*H*U ?= H)
    '''
    
    hels3D =  make_sign_matrix(Lm ,Wg)
    signmat = kwant.qsymm.builder_to_model(hels3D)[1]  
    
    return signmat



def make_sign_matrix_prime(Lm ,Wg):
    a   = 1
    lat =  kwant.lattice.cubic(a,norbs = 2) # 2 d.o.f. per site => norbs = 2
    syst = kwant.Builder()        
    
    d = Wg
    for i in range( -Lm , Lm + 1  ):
        for j in range(-d, d):
            
            
           
            # Sgeom is created:     
            if 0 <= i <= Lm and 0 <= j  <= Wg:
                syst[lat(i, j,0)] =  ((-1)**(i+j))*sigma_x
            if -Lm+1 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst[lat(i, j,0)] = ((-1)**(i+j))*sigma_x
              
           
  
    
    return syst


def hamiltonian_signmatrix_prime(Lm,Wg):
    '''
    Input:
    - Lm , Wg = geometric paramteters of of the S geometry
    Output:
    - Diagonal matrix U, which has +1 on sublattice A and -1 on sublattice B. (Matrix can used to verify -U*H*U ?= H)
    '''
    
    hels3D =  make_sign_matrix_prime(Lm ,Wg)
    signmat = kwant.qsymm.builder_to_model(hels3D)[1]  
    
    return signmat



########################################################################################################################
### Scattering region where spin-orbit coupling is generated by field in x,z direction.
########################################################################################################################



def make_system_wbl_ezvec(t,tsom,Lm ,Wg,depth=0):
    
    
    
    
    #Define hoppings in z,x direction
    # The z - direction lies along lead direction
    # The x - direction is defined in-plane orthogonal to the z-direction
    # The y - direction points out-of-plane
    # Thus the lattice has coordinate label (z,x,y) (instead of the usual (x,y,z))
    
    # Create Lattice
    
    a   = 1
    lat =  kwant.lattice.cubic(a,norbs = 2) # 2 d.o.f. per site => norbs = 2
    syst = kwant.Builder()
    
    U = 0        #onsite energy
          #molecule length
    
     ### DEFINE LATTICE HAMILTONIAN ###
        
    
    d = Wg
    
    for K in range(0,depth+1):
        for i in range( -Lm , Lm + 1  ):
            for j in range(-d, d):



                # Sgeom is created:     
                if 0 <= i <= Lm and 0 <= j  <= Wg:
                    syst[lat(i, j,K)] =  U*sigma_0
                if -Lm+1 <= i <= 1 and -Wg+1 <= j  <= 0:
                    syst[lat(i, j,K)] = U*sigma_0
              
           
    # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0,0), lat, lat)] =  -t*sigma_0 - 1j*tsom*sigma_x
    #hopping in x direction
    syst[kwant.builder.HoppingKind((0,1,0), lat, lat)] =  -t*sigma_0 
    
    #hopping in y direction
#     syst[kwant.builder.HoppingKind((0,0,1), lat, lat)] =  -t*sigma_0 + 1j* tsom* sigma_x

    
    return syst

def make_system_wbl_exvec(t,tsom,Lm ,Wg):
    
    
    
    
    #Define hoppings in z,x direction
    # The z - direction lies along lead direction
    # The x - direction is defined in-plane orthogonal to the z-direction
    # The y - direction points out-of-plane
    # Thus the lattice has coordinate label (z,x,y) (instead of the usual (x,y,z))
    
    # Create Lattice
    
    a   = 1
    lat =  kwant.lattice.cubic(a,norbs = 2) # 2 d.o.f. per site => norbs = 2
    syst = kwant.Builder()
    
    U = 0        #onsite energy
          #molecule length
    
     ### DEFINE LATTICE HAMILTONIAN ###
        
    
    d = Wg
    
    for i in range( -Lm , Lm + 1  ):
        for j in range(-d, d):



            # Sgeom is created:     
            if 0 <= i <= Lm and 0 <= j  <= Wg:
                syst[lat(i, j,0)] =  U*sigma_0
            if -Lm+1 <= i <= 1 and -Wg+1 <= j  <= 0:
                syst[lat(i, j,0)] = U*sigma_0
              
           
    # hopping in z-direction
    syst[kwant.builder.HoppingKind((1,0,0), lat, lat)] =  -t*sigma_0 + 1j*tsom*sigma_y
    #hopping in x direction
    syst[kwant.builder.HoppingKind((0,1,0), lat, lat)] =  -t*sigma_0 
    
    

    
    return syst


def hamiltonian_multiplesites_ez(L,Wg, t,tsom,gamma,pz,gammaB,depth=0,plotbool=False):
    

    
    
    hels3D =  make_system_wbl_ezvec(t,tsom,L ,Wg,depth) 
    
    
    if plotbool == True:
        kwant.plot(hels3D);
    
    Hamiltonian0 = kwant.qsymm.builder_to_model(hels3D)[1]
    
    shape_gammas = Hamiltonian0.shape
   
    GammaL = np.zeros(shape_gammas)
    GammaRP = np.zeros(shape_gammas)
    GammaRM = np.zeros(shape_gammas)
    
    GammaB = np.zeros(shape_gammas)

   
    for j in range(0,Wg):
        GammaL[2*j:2*(j+1),2*j:2*(j+1)] = np.multiply(gamma,sigma_0)
#         GammaL[2:4,2:4] = np.multiply(gamma,sigma_0)
    

        GammaRP[shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j,shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j] = gamma*sigma_0 + gamma*abs(pz)*sigma_z
        
        GammaRM[shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j,shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j] = gamma*sigma_0 -gamma*abs(pz)*sigma_z
#         GammaRP[shape_gammas[0]-4:shape_gammas[0]-2,shape_gammas[0]-4:shape_gammas[0]-2] = gamma*sigma_0 + gamma*abs(pz)*sigma_z
    
    
#     GammaRM[shape_gammas[0]-2:shape_gammas[0],shape_gammas[0]-2:shape_gammas[0]] = gamma*sigma_0 - gamma*abs(pz)*sigma_z
#     GammaRM[shape_gammas[0]-4:shape_gammas[0]-2,shape_gammas[0]-4:shape_gammas[0]-2] = gamma*sigma_0 - gamma*abs(pz)*sigma_z
   
    
    

    return Hamiltonian0,GammaL,GammaRP,GammaRM
 

def hamiltonian_multiplesites_ex(L,Wg, t,tsom,gamma,pz,plotbool=False):
    

    
    
    hels3D =  make_system_wbl_exvec(t,tsom,L ,Wg) 
    
    
    if plotbool == True:
        kwant.plot(hels3D);
    
    Hamiltonian0 = kwant.qsymm.builder_to_model(hels3D)[1]
    
    shape_gammas = Hamiltonian0.shape
   
    GammaL = np.zeros(shape_gammas)
    GammaRP = np.zeros(shape_gammas)
    GammaRM = np.zeros(shape_gammas)
    
    GammaB = np.zeros(shape_gammas)

   
    for j in range(0,Wg):
        GammaL[2*j:2*(j+1),2*j:2*(j+1)] = np.multiply(gamma,sigma_0)
#         GammaL[2:4,2:4] = np.multiply(gamma,sigma_0)
    

        GammaRP[shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j,shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j] = gamma*sigma_0 + gamma*abs(pz)*sigma_z
        
        GammaRM[shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j,shape_gammas[0]-2*(j+1):shape_gammas[0]-2*j] = gamma*sigma_0 -gamma*abs(pz)*sigma_z

    

    return Hamiltonian0,GammaL,GammaRP,GammaRM
