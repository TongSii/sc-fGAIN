#Imports
import time
import pandas as pd
import numpy as np
from utils import *
from fGAIN_mask import gain

import argparse
import tensorflow.compat.v1 as tf

def main(data, miss_rate, batch_size, hint_rate, alpha, iterations, loss_fn):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data: pandas dataframe of the data
    - mask: a pandas dataframe with 1,0's
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''

  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': iterations} 
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m,MASK = pandas_data_loader(data, miss_rate)
  #df1 = data.astype(bool).astype(int)
  #df1.columns = data.columns
  #MASK = df1


  imputed_data_x = gain(MASK,miss_data_x, gain_parameters, loss_fn=loss_fn)

  # Report the RMSE performance
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  SCC= scc(ori_data_x, imputed_data_x, data_m) 
  SCC = pd.DataFrame(SCC)
  #print()
  #print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x, rmse,SCC

if __name__ == "__main__":
    #Set of starting variables
    dataFPath = ""
    loss_func = ""
    #miss_rate = .2
    #batch_size = 60
    batch_size = 128
    hint_rate = .9
    #alpha = 100
    alpha = 100
    iterations = 10000
    #get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataFile", help="Uses the given data at the file path")
    parser.add_argument("-m", "--missRate", help="the missing rate")
    parser.add_argument("-l", "--lossFn", help="Uses what loss function is given")
    
    
    #read arguments
    args = parser.parse_args()
    dataFPath = args.dataFile
    loss_func = args.lossFn
    missing_rate = float(args.missRate)
    
    #trys to read the data file
    try:
        x_df = pd.read_csv(dataFPath, delimiter=',',header= 0,index_col = 0 )
        #x_df_1 = x_df.iloc[:,1:50]
        #x_df_204 = x_df.iloc[:,10150:10166]
        #x_df =  x_df.transpose()
        #x_df = x_df.T.reset_index(drop=True)
        #x_df1 = x_df.iloc[:,0:100]
        #x_df2 = x_df.iloc[:,100:200]
        #x_df1 = x_df[x_df.columns[1:2000]]
        #x_df3 = x_df.iloc[:,200:300]
        #x_df4 = x_df.iloc[:,300:400]
        #x_df5 = x_df.iloc[:,400:500]
        #x_df6 = x_df.iloc[:,500:600]
        #x_df7 = x_df.iloc[:,600:700]
        #x_df8 = x_df.iloc[:,700:800]
        #x_df9 = x_df.iloc[:,800:900]
        #x_df10 = x_df.iloc[:,900:1000]
        #x_df11 = x_df.iloc[:,900:1000]
        #x_df12 = x_df.iloc[:,900:1000]
        #x_df13 = x_df.iloc[:,900:1000]
        #x_df14 = x_df.iloc[:,900:1000]
        #x_df15 = x_df.iloc[:,900:1000]
        #x_df16 = x_df.iloc[:,900:1000]
        #x_df17 = x_df.iloc[:,900:1000]
        #x_df18 = x_df.iloc[:,900:1000]
        #x_df19 = x_df.iloc[:,900:1000]
        #x_df20 = x_df.iloc[:,900:1000]
        #x_df21 = x_df.iloc[:,900:1000]
        #x_df22 = x_df.iloc[:,900:1000]
        #x_df23 = x_df.iloc[:,900:1000]
       # for i in range(2,102):
       #     x_df_i = x_df.iloc[:,100*(i-1):100*i]
            #return x_df_i
        #x_df_1 = x_df.iloc[:,1:100]
        #x_df_102 = x_df.iloc[:,10100:10166]


       #x_df=[x_df_1,x_df_2,x_df_3,x_df_4,x_df_5]
    except FileNotFoundError:
        print("Can't find File {0}".format(dataFPath))
        exit(1)
    
    #runs the main function of the gain
    #i=1
    #for i in range(1,103):
    #    imputated_data_i, rmse_i = main(x_df_i, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
        #return imputated_data_i, rmse_i
    imputated_data, rmse,SCC = main(x_df, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data_1, rmse_1 = main(x_df_1, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data1, rmse1 = main(x_df1, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data2 , rmse2 = main(x_df2, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data3 , rmse3 = main(x_df3, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    
    #imputated_data4 , rmse4 = main(x_df4, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    
    #imputated_data5 , rmse5 = main(x_df5, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_dat6 , rmse6 = main(x_df6, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data7 , rmse7 = main(x_df7, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data8 , rmse8 = main(x_df8, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data9 , rmse9 = main(x_df9, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)
    #imputated_data10 , rmse10 = main(x_df10, missing_rate, batch_size, hint_rate, alpha, iterations, loss_func)


#imputated_data = [imputated_data1,imputated_data2,imputated_data3,imputated_data4,imputated_data5]
    #saves the imputated data out to a file with the name of the loss function 
    #outputFileName = "/data/projects/zackary.hopkins/imputated_Data/{0}_{1}_imputated_data.csv".format(dataFPath[-18:-4], loss_func)
#    for i in range(1,103):
 #       outputFileName_i = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data_norm_{2}.csv".format(dataFPath[-18:-4], loss_func,i)
  #      np.savetxt(outputFileName_i, imputated_data_i, delimiter=",")
        #return outputFileName_i 

    #outputFileName1 = "/student/tsi2/r-venv/ra-code/imputated_Data_norm_3/imputated_data_norm_102.csv".format(dataFPath[-18:-4], loss_func)
    outputFileName = "/student/tsi2/r-venv/ra-code/missing_all/{0}_{1}imputated_data_norm_newmask.csv".format(dataFPath[-11:-4], loss_func)
    #outputFileName1 = "/student/tsi2/r-venv/ra-code/cg_all_norm_log2_divide/original_GAIN/imputated_data_norm_1.csv"

    #np.savetxt(outputFileName204, imputated_data_204, delimiter=",")
    np.savetxt(outputFileName, imputated_data, delimiter=",")
    #SCC.to_csv("/student/tsi2/r-venv/ra-code/cg_all_DE_norm_log2_divide/SCC_newmask.csv")
    #outputFileName2 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_2.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName2, imputated_data2, delimiter=",")
     
    #outputFileName3 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_3.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName3, imputated_data3, delimiter=",")
     
    #outputFileName4 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_4.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName4, imputated_data4, delimiter=",")
     
    #outputFileName5 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_5.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName5, imputated_data5, delimiter=",")
    #outputFileName6 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_6.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName6, imputated_data5, delimiter=",")
    #outputFileName7 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_7.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName7, imputated_data5, delimiter=",")
    #outputFileName8 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_8.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName8, imputated_data5, delimiter=",")
    #outputFileName9 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_9.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName9, imputated_data5, delimiter=",")
    #outputFileName10 = "/student/tsi2/r-venv/ra-code/imputated_Data/{0}_{1}_imputated_data2_10.csv".format(dataFPath[-18:-4], loss_func)
    #np.savetxt(outputFileName10, imputated_data5, delimiter=",")
    #outputs the results
    #print("The imputated data 1 has been saved to {0} RMSE: {1}".format(outputFileName1, rmse1))
    #print("The imputated data i has been saved to {0} RMSE: {1}".format(outputFileName2, rmse2))
    #print("The imputated data i has been saved to {0} RMSE: {1}".format(outputFileName3, rmse3))
    #print("The imputated data i has been saved to {0} RMSE: {1}".format(outputFileName4, rmse4))
    print(rmse)
