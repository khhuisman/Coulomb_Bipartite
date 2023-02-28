#Author: Karssien Hero Huisman

#Notebook for functions that appear throughout both the Hartree-Fock and Hubbard-One notebooks.
import numpy as np



# In[5]:

def jointwolist(a,b):
    '''
    Input: list a and list b.
    Output: Two joined lists.
    '''
    ab_list = []
    for element in a:
        ab_list.append(element)
    
    for element in b:
        ab_list.append(element)
        
    return ab_list


def check_listzero(alist,tol):

    '''
    Input
    - two lists: lista,listb
    - tol = convergence criterium
    Ouput:
    check_zero = absolute value of the difference between lista,listb rounded by tol
    zero_bool = True if convergence is achieved'''
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    check_zero = np.round(alist,acc)

    dim = len(check_zero)
    teller = 0


    for element in check_zero:


        if element == 0.0:
            teller +=  1

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return check_zero, zero_bool        
        
        
def check_difference(lista,listb,tol):

    '''
    Input
    - two lists: lista,listb
    - tol = convergence criterium
    Ouput:
    check_zero = absolute value of the difference between lista,listb rounded by tol
    zero_bool = True if convergence is achieved'''
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    check_zero = abs(np.round(np.subtract(lista,listb),acc))


    dim = len(check_zero)
    teller = 0


    for element in check_zero:


        if element == 0.0:
            teller +=  1

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return check_zero, zero_bool
            


def func_list_i(Hamiltonian):
    '''
    Input:
    -  N X N Hamiltonian
    Output:
    list of labels corresponding the the shape of the shape: [0,1,2,...,N-1]
    '''
    number_sites_spin,number_sites_spin = Hamiltonian.shape
    
    ilist = [i   for i in range(number_sites_spin)]
    
    return ilist

def check_difference2(lista,listb,tol):

    '''
    Input
    - two lists: lista,listb
    - tol = convergence criterium
    Ouput:
    check_zero_prune = absolute value of the difference between lista,listb subtracted by tol
    zero_bool = True if convergence is achieved'''
    check_zero = abs(np.subtract(lista,listb))


    dim = len(check_zero)
    teller = 0

    check_zeroprime = []
    for element in check_zero:


        if element <= tol:
            teller +=  1
            check_zeroprime.append(0)
        if element > tol:
            check_zeroprime.append(element)
        

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return check_zeroprime, zero_bool


def halves_list(Hamiltonian):
    
    ''' Input:
    - N X N Hamiltonian
    Output:
    - list: = [1/2,1/2,....] length of the list is N.
    '''
    
    number_sites_spin,number_sites_spin = Hamiltonian.shape

    n_list_s = [0.5   for i in range(number_sites_spin)]
    
    return n_list_s

def list_halves(Hamiltonian):
    
    ''' Input:
    - N X N Hamiltonian
    Output:
    - list: = [1/2,1/2,....] length of the list is N.
    '''
    
    number_sites_spin,number_sites_spin = Hamiltonian.shape

    n_list_s = [0.5   for i in range(number_sites_spin)]
    
    return n_list_s

# In[49]:

def pairwise_swap(xlist):
    '''
    Input:
    list with elements : [a1,a2,b1,b2,...]
    Ouput:
    list where every 2 elements are swapped: [a2,a1,b2,b1,...]
    '''
    
    xlist_swapped = []
    
    for i in range(0,len(xlist),2):
        element_even = xlist[i]
        element_odd = xlist[i+1]
        
        xlist_swapped.append(element_odd)
        xlist_swapped.append(element_even)
    
    return xlist_swapped

def func_V_total(Vmax,dV):
    
    '''
    Input: 
    Vmax = maximum bias voltage
    dV   = voltage stepsize
    Output: 
    V_list_pos_bias : list with only positive voltages
    V_list_total    : list off all voltages
    '''
    
    V_list_pos_bias = np.arange(dV,Vmax + dV,dV)

    V_list_neg_bias = -1*np.flip(V_list_pos_bias)

    V_list_total = jointwolist(jointwolist(V_list_neg_bias,[0]),V_list_pos_bias)

    return V_list_pos_bias,V_list_total

def check_list_smaller(alist,value):

    '''
    Input
    - alist = list with values
    - value = positive number
    Ouput:
    zero_bool = True if values in alist are smaller than 'value' and bigger or equal to zero:
                0 <= alist[i] <= value '''

    dim = len(alist)
    teller = 0


    for element in alist:


        if element <= value and element > 0:
            teller +=  1

    if teller == dim:
        zero_bool = True

    if teller != dim:
        zero_bool = False
        
    return zero_bool

#######################################################################################################################
############################################# assymetry factor ########################################################
#######################################################################################################################

def func_chi(a,b):
    return (a-b)/(a+b)
        

#######################################################################################################################
############################################# check convergence #######################################################
#######################################################################################################################


def converged_lists(V_list_total,
                      nP_list_total ,convglistP,
                    nM_list_total, convglistM):
    
    '''Input:
    - V_list_total                = list of all bias voltages considered
    - nP_list_total,nP_list_total = electron densities for positive,negative magnetization as function of bias voltage
    - convglistP,convglistM       = list of booleans as function of voltage for positive,negative magnetization indicating 
                                    if density converged ('True') or did not converge ('False')
    Output
    - list of electron densities for positive and negative magnetization that both have converged \
    for the corresponding voltages'''
    
   
    nP_list_combined = []
    nM_list_combined = []
    V_list_combined = []

    for i in range(len(V_list_total)):
        V = V_list_total[i]
        boolP = convglistP[i]
        boolM = convglistM[i]
        
        if boolP == True and boolM == True:
            V_list_combined.append(V)
            nP_list_combined.append(nP_list_total[i])
            nM_list_combined.append(nM_list_total[i])
            
            
    return V_list_combined,nP_list_combined,nM_list_combined  


#######################################################################################################################
############################################# filter positive,negative bias ###########################################
#######################################################################################################################

def evenodd_index_list(xlist):
    
    '''
    Input: 
    List of ascending real numbers. 
    The first and last element are negative and postive respectively  (x0 < 0 and xn > 0):
    
        xlist  = [x0,x1,x2,...,xn]
        
    Output: 
    List of labels 'k' s.t. for the element xj in the list xlist we have:
            xlist[k] + xj = 0.
    If no such label exist, then nothing is returned.
    '''

    index_sol_list = []
    lenx = len(xlist)
    for n in range(len(xlist)):
        xj      = xlist[-1-n] 
        xlist_j = [ xj for element in range(lenx) ]
        
        sol_x = np.where(np.round(np.add(xlist,xlist_j),11) == 0)[0] #search index s.t. xn + xlist[k] = 0
        

        if len(sol_x) == 1:
            index_sol_list.append(sol_x[0])
       
    return index_sol_list  


def func_symmetric_list(V_list_converged,
                      nP_list_V ,
                    nM_list_V):
    
    '''Input:
    - V_list_converged    = list of converged bias voltages
    - nP_list_V,nM_list_V = converged electron densities for both positive,negative magnetization as function of bias voltage
    Output
    - List of voltages such that:
        V_list_converged + np.flip(V_list_converged) = [0,0,...]
        
    - List of electron densities for positive,negative magnetization s.t. 
      there are two element for both positive AND negative voltage (except for V=0 where there is 1 element): 
        
        [n(-V),n(-V + dV), ... , n(-dV), n(0),n(dV), ..... n(V - dV), n(V)]
        
    '''
    
    indexlist = evenodd_index_list(V_list_converged)
    V_list =  [ V_list_converged[index] for index in indexlist]
    nP_list = [ nP_list_V[index] for index in indexlist]
    nM_list = [ nM_list_V[index] for index in indexlist]
 
    return V_list,nP_list,nM_list  


#######################################################################################################################
################################# combine convergence & filter in bias voltage ########################################
#######################################################################################################################


def func_symmetric_converged(V_list_total,
                              nP_list_total ,convglistP,
                              nM_list_total, convglistM):
    
    '''Input:
    - V_list_total                = list of all bias voltages considered
    - nP_list_total,nP_list_total = electron densities for positive,negative magnetization as function of bias voltage
    - convglistP,convglistM       = list of booleans as function of voltage for positive,negative magnetization indicating 
                                    if density converged ('True') or did not converge ('False')
    Output
    - List of voltages such that:
        V_list_converged + np.flip(V_list_converged) = [0,0,...]
        
    - List of electron densities for positive,negative magnetization s.t. 
      there are two element for both positive AND negative voltage (except for V=0 where there is 1 element): 
        
        [n(-V),n(-V + dV), ... , n(-dV), n(0),n(dV), ..... n(V - dV), n(V)]
        
    '''
    
    V_list_combined,nP_list_combined,nM_list_combined  = converged_lists(V_list_total,
                                                                        nP_list_total ,convglistP,
                                                                        nM_list_total, convglistM)
    
    
    V_list,nP_list,nM_list  = func_symmetric_list(V_list_combined,nP_list_combined ,nM_list_combined)
    
    return V_list,nP_list,nM_list










