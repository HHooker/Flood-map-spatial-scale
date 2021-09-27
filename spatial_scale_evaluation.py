# FSS Functions

import numpy as np
import matplotlib.pyplot as plt
# import ArrayValues

def binary_map(input_file):
    """Read ascii file into an array, remove header rows, convert to binary 
    grid unflooded 0, flooded 1"""
    
    ascii_grid = np.loadtxt(input_file, skiprows=6)
    ascii_zeros = np.where(ascii_grid<=0, 0, ascii_grid)
    ascii_binary = np.where(ascii_zeros>0, 1, ascii_zeros)
    
    return(ascii_binary)

def flood_depth_map(input_file):
    """Read ascii file into an array, remove header rows, convert to 
    unflooded 0, flooded depths 1, 2, 3, and 4"""
    
    ascii_grid = np.loadtxt(input_file, skiprows=6)
    ascii_depth = np.where(ascii_grid<=0, 0, ascii_grid)
    
    return(ascii_depth)

def flood_edge_map(input_file):
    """Read ascii file into an array, remove header rows, convert to 
    unflooded 0, and only flooded depths 1 and 2"""
    
    ascii_grid = np.loadtxt(input_file, skiprows=6)
    ascii_depth = np.where(ascii_grid<=0, 0, ascii_grid)
    ascii_depthb = np.where(ascii_depth==2, 1, ascii_depth)
    ascii_edge = np.where(ascii_depthb>1, 0, ascii_depthb)
    
    return(ascii_edge)

def flood_count(binary_map):
    """Counts the number of flooded grid cells"""
    
    flood_count = (binary_map==1).sum()
     
    return(flood_count)
     
def depth_count(depth_map):
    """Creates an array of the count and proportion of flood depths from the 
    depth map"""
    
    depth_count_0 = (depth_map==0).sum()
    depth_count_1 = (depth_map==1).sum()
    depth_count_2 = (depth_map==2).sum()
    depth_count_3 = (depth_map==3).sum()
    depth_count_4 = (depth_map==4).sum()
    
    depth_count = [depth_count_0, depth_count_1, depth_count_2, depth_count_3, \
                   depth_count_4]
    depth_count_total = np.sum(depth_count)
    depth_percent = np.round((depth_count/depth_count_total)*100, 4)
    
    
    depth_group = [0.0, 1.0, 2.0, 3.0, 4.0]
    depth_group_tuple = (depth_group, depth_count, depth_percent)
    depth_group_table = np.transpose(np.vstack(depth_group_tuple))
    
    return(depth_group_table)


def fraction_calc(input_array, l, k, n):
    """ calculate the fraction flooded at  one location i, j for a neighbourhood
    size n x n"""

    fraction = np.mean(input_array[l - ((n - 1) // 2): l + ((n - 1) // 2) + 1, k - ((n - 1) // 2): k + ((n - 1) // 2) + 1])

    return fraction



def fraction_field(flood_map, max_n, n):
    """ Create a field of fractions from a binary flood map across a domain
        of interest with max_n neighbourhood size and neighbourhood of interest n"""
    
    num_rows = np.shape(flood_map)[0]
    num_columns = np.shape(flood_map)[1]
    row_start = int((max_n-1)/2)
    row_stop = int(num_rows-((max_n-1)/2))
    column_start = int((max_n-1)/2)
    column_stop = int(num_columns-((max_n-1)/2))
    domain = flood_map[row_start:row_stop,column_start:column_stop]
    domain_rows = np.shape(domain)[0]
    domain_columns = np.shape(domain)[1]
    
    #array_values = ArrayValues(domain, row_start, row_end, column_start, column_end)
    
    fraction_array = np.zeros((domain_rows,domain_columns))
    
    for i in range(row_start, row_stop):
        for j in range(column_start, column_stop):
            fraction_array[i-row_start,j-column_start] = fraction_calc(flood_map, i, j, n)

    return(fraction_array)    


def MSE(fraction_array_o, fraction_array_m):
    """ return the average mean squared error between 2 arrays"""
    
    MSE_array = np.square(fraction_array_o-fraction_array_m)
    MSE_ave = np.mean(MSE_array)
    
    return(MSE_ave)

def MSE_ref(fraction_array_o, fraction_array_m):
    """"return the reference average MSE between 2 arrays"""
    
    MSE_ref_array = np.square(fraction_array_o)+np.square(fraction_array_m)
    MSE_ref_ave = np.mean(MSE_ref_array)

    return(MSE_ref_ave)

def FSS_table(input_array_o, input_array_m, maxn):
    """ For 2 input binary arrays, calculate the FSS for neighbourhoods size n up
    to a maxn (ODD) and produce a table of FSS for each n and plot n vs FSS"""
    
    test_size = int((maxn+2)/2)
    FSS_n = np.zeros(test_size)
    
    for neigh in range(1, maxn+1, 2):
    
        fraction_array_test_o = fraction_field(input_array_o, maxn, neigh)
        fraction_array_test_m = fraction_field(input_array_m, maxn, neigh)
        MSE_o_m = MSE(fraction_array_test_o, fraction_array_test_m)
        MSE_o_m_ref = MSE_ref(fraction_array_test_o, fraction_array_test_m)
        
        FSS_n[int((neigh-1)/2)] = 1-(MSE_o_m/MSE_o_m_ref)  

    n_column = np.arange(1, maxn+1, 2)
    FSS_table_tuple = (n_column, FSS_n)
    FSS_table = np.transpose(np.vstack(FSS_table_tuple))

    return(FSS_table)

def FSS_plot(FSS_results_table):
    
    plt.figure(1)
    plt.plot(FSS_results_table[:,0], FSS_results_table[:,1])
    plt.ylabel("FSS")
    plt.xlabel("neighbourhood size n")
    plt.title("FSS vs neighbourhood size")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=0)
    plt.legend()
    plt.show() 

def contingency_array(model_map, obs_map):
    """Create a contingency map A, B, C, D from a binary model and observed flood map"""
    
    n_rows = np.shape(model_map)[0]
    n_columns = np.shape(model_map)[1]
    
    cont_array = np.zeros((n_rows, n_columns))
    
    cont_A = np.where((model_map==1) & (obs_map==1), 'A', cont_array)
    cont_B = np.where((model_map==1) & (obs_map==0), 'B', cont_A)
    cont_C = np.where((model_map==0) & (obs_map==1), 'C', cont_B)       
    cont_all = np.where((model_map==0) & (obs_map==0), 'D', cont_C)
       
    return(cont_all)   

def contingency_array_numerical(model_map, obs_map):
    """Create a contingency map 1, 2, 3, 4 from a binary model and observed flood map"""
    
    n_rows = np.shape(model_map)[0]
    n_columns = np.shape(model_map)[1]
    
    cont_array = np.zeros((n_rows, n_columns))
    
    cont_A = np.where((model_map==1) & (obs_map==1), 1.0, cont_array)
    cont_B = np.where((model_map==1) & (obs_map==0), 2.0, cont_A)
    cont_C = np.where((model_map==0) & (obs_map==1), 3.0, cont_B)       
    cont_all_numerical = np.where((model_map==0) & (obs_map==0), 4.0, cont_C)
       
    return(cont_all_numerical)  

def agreement_field_D(depth_map_1, depth_map_2):
    
    no_rows = np.shape(depth_map_1)[0]
    no_columns = np.shape(depth_map_1)[1]
    agreement_field = np.zeros((no_rows, no_columns))
    
    for i in range(0, no_rows):
        for j in range(0, no_columns):
            
            if depth_map_1[i,j]==0 and depth_map_2[i,j]==0: 
                agreement_field[i,j] = 0.0
            
            elif (depth_map_1[i,j]-depth_map_2[i,j])==0:
                agreement_field[i,j] = 0.0          
            
            else:
                agreement_field[i,j] = np.square(depth_map_1[i,j]-depth_map_2[i,j]) \
                /(np.square(depth_map_1[i,j])+np.square(depth_map_2[i,j]))

    return(agreement_field)


def scale_field(depth_map_1, depth_map_2, alpha, nmax):
    
    num_rows = np.shape(depth_map_1)[0]
    num_columns = np.shape(depth_map_1)[1]
    row_start = int((nmax-1)/2)
    row_stop = int(num_rows-((nmax-1)/2))
    column_start = int((nmax-1)/2)
    column_stop = int(num_columns-((nmax-1)/2))
    domain = depth_map_1[row_start:row_stop,column_start:column_stop]
    domain_rows = np.shape(domain)[0]
    domain_columns = np.shape(domain)[1]
    

    scale_array = np.zeros((domain_rows,domain_columns))
    smax = (nmax-1)/2
    smax = int(smax)
    agreement_field = np.ones((domain_rows,domain_columns))
      
    for i in range(row_start, row_stop):
        for j in range(column_start, column_stop):
                
                S = 0
                if depth_map_1[i,j]==0 and depth_map_2[i,j]==0: 
                    agreement_field[i-row_start,j-column_start] = 0.0

                elif (depth_map_1[i,j]-depth_map_2[i,j])==0:
                    agreement_field[i-row_start,j-column_start] = 0.0          
            
                else:
                    agreement_field[i-row_start,j-column_start] = np.square(depth_map_1[i,j]-depth_map_2[i,j]) \
                        /(np.square(depth_map_1[i,j])+np.square(depth_map_2[i,j]))
                
                Dcrit = alpha + (1-alpha)*(S/smax)
                
                if agreement_field[i-row_start,j-column_start] == 0.0:
                    scale_array[i-row_start,j-column_start] = 0.0
                    
                elif agreement_field[i-row_start,j-column_start] <= Dcrit:
                    scale_array[i-row_start,j-column_start] = S
                
                else:
                    for S in range(1, smax+1):
                        
                        n = (2 * S) + 1
                              
                        if (fraction_calc(depth_map_1, i, j, n))-(fraction_calc(depth_map_2, i, j, n))==0:
                            agreement_field[i-row_start,j-column_start] = 0.0          
            
                        else:
                            agreement_field[i-row_start,j-column_start] = np.square((fraction_calc(depth_map_1, i, j, n))-(fraction_calc(depth_map_2, i, j, n))) \
                                /(np.square(fraction_calc(depth_map_1, i, j, n))+np.square(fraction_calc(depth_map_2, i, j, n)))
                
                        Dcrit = alpha + (1-alpha)*(S/smax)
                
                        if agreement_field[i-row_start,j-column_start] <= Dcrit:
                            scale_array[i-row_start,j-column_start] = S
                            break                     
    return(scale_array) 

def FSS_target(observed_flood):
    """calculate the FSS target score for a given observed flood map"""
    
    obs_flood_count = flood_count(observed_flood)
    array_size = np.size(observed_flood)
    obs_flood_fraction = obs_flood_count/array_size
    FSS_target = 0.5 + (obs_flood_fraction/2)
    
    return(FSS_target)

def asymptote_FSS(observed_flood, model_flood):
    """calculate the asymptote AFSS limit for a given observed flood map"""
    
    o_flood_count = flood_count(observed_flood)
    m_flood_count = flood_count(model_flood)
    array_size_o = np.size(observed_flood)
    o_flood_fraction = o_flood_count/array_size_o
    m_flood_fraction = m_flood_count/array_size_o
    
    asymptote_FFS = (2 * o_flood_fraction * m_flood_fraction)/((o_flood_fraction*o_flood_fraction) + (m_flood_fraction*m_flood_fraction))
    
    return(asymptote_FFS)

def scale_field_f(depth_map_1, depth_map_2, alpha, nmax):
    
    num_rows = np.shape(depth_map_1)[0]
    num_columns = np.shape(depth_map_1)[1]
    row_start = int((nmax-1)/2)
    row_stop = int(num_rows-((nmax-1)/2))
    column_start = int((nmax-1)/2)
    column_stop = int(num_columns-((nmax-1)/2))
    domain = depth_map_1[row_start:row_stop,column_start:column_stop]
    domain_rows = np.shape(domain)[0]
    domain_columns = np.shape(domain)[1]
    

    scale_array = np.zeros((domain_rows,domain_columns))
    smax = (nmax-1)/2
    smax = int(smax)
    agreement_field = np.ones((domain_rows,domain_columns))        
    S = 0
    
    if depth_map_1[row_start:row_stop,column_start:column_stop]==0 and depth_map_2[row_start:row_stop,column_start:column_stop]==0: 
        agreement_field = 0.0

    elif (depth_map_1[row_start:row_stop,column_start:column_stop]-depth_map_2[row_start:row_stop,column_start:column_stop])==0:
        agreement_field = 0.0          
            
    else:
        agreement_field = np.square(depth_map_1[row_start:row_stop,column_start:column_stop]-depth_map_2[row_start:row_stop,column_start:column_stop]) \
            /(np.square(depth_map_1[row_start:row_stop,column_start:column_stop])+np.square(depth_map_2[row_start:row_stop,column_start:column_stop]))
                

    for S in range(1, smax+1):
                        
        n = (2 * S) + 1
                              
        if (np.mean(depth_map_1[((n - 1) // 2):((n - 1) // 2) + 1,((n - 1) // 2):((n - 1) // 2) + 1]) - np.mean(depth_map_2[((n - 1) // 2):((n - 1) // 2) + 1,((n - 1) // 2):((n - 1) // 2) + 1]))==0:
            agreement_field = 0.0          
            
        else:
            agreement_field = np.square((np.mean(depth_map_1[((n - 1) // 2):((n - 1) // 2) + 1,((n - 1) // 2):((n - 1) // 2) + 1]) - np.mean(depth_map_2[((n - 1) // 2):((n - 1) // 2) + 1,((n - 1) // 2):((n - 1) // 2) + 1]))) \
                /(np.square(np.mean(depth_map_1[((n - 1) // 2):((n - 1) // 2) + 1,((n - 1) // 2):((n - 1) // 2) + 1])) + np.square(np.mean(depth_map_2[((n - 1) // 2):((n - 1) // 2) + 1,((n - 1) // 2):((n - 1) // 2) + 1])))
                
        Dcrit = alpha + (1-alpha)*(S/smax)
                
        if agreement_field <= Dcrit:
            scale_array = S
            break    

    return(scale_array) 


































