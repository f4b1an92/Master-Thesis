import os
import sys
import yaml 
import runpy
import time

os.path.realpath(__file__)
main_path = os.getcwd()

   
# Cretate Directories
# cleaned_datasets
if "cleaned_datasets" in os.listdir():
    pass
else:
    os.mkdir("cleaned_datasets")

# models
if "models" in os.listdir():
    pass
else:
    os.mkdir("models")
    os.mkdir("models\\trainers")

# results
if "results" in os.listdir():
    pass
else:
    os.mkdir("results")
    os.mkdir("results\\FL Results")
    os.mkdir("results\\Non-FL Results")
    os.mkdir("results\\Plot Data")
    os.mkdir("results\\Non-FL Results\\Best Params Horizontal")
    os.mkdir("results\\Non-FL Results\\Best Params Vertical")

code_path = main_path + '\\code' 
start_overall = time.time()


# 1. Create Data
#=======================
start = time.time()
runpy.run_path(path_name=code_path + '\\data_cleaning+storing.py')
end = time.time()

print(80 * '=')
print(f"\n\t Data cleaning and feature engineering: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 2. Parameter Tuning for horizotntal federated learning
#=========================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\horizontal_nfm.py')
end = time.time()

print(80 * '=')
print(f"\n\t Tuning of Non-federated models on horizontal data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 3. Parameter Tuning for vertical federated learning
#=========================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\vertical_nfm.py')
end = time.time()

print(80 * '=')
print(f"\n\t Tuning of Non-federated models on vertical data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 4. Fitting federated of log. regression on horizontal data
#==============================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\horizontal_fm_LG.py')
end = time.time()

print(80 * '=')
print(f"\n\t Fitting of federated log. regression on horizontal data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 5. Fitting federated of neural networks on horizontal data
#================================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\horizontal_fm_NN.py')
end = time.time()

print(80 * '=')
print(f"\n\t Fitting of federated neural networks on horizontal data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 6. Fitting federated forest on horizontal data
#====================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\horizontal_fm_FF.py')
end = time.time()

print(80 * '=')
print(f"\n\t Fitting of federated forest on horizontal data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 7. Fitting federated of log. regression on vertical data
#==============================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\vertical_fm_LG.py')
end = time.time()

print(80 * '=')
print(f"\n\t Fitting of federated log. regression on vertical data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 8. Fitting federated of neural networks on vertical data
#================================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\vertical_fm_NN.py')
end = time.time()

print(80 * '=')
print(f"\n\t Fitting of federated neural networks on vertical data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 9. Fitting federated forest on vertical data
#====================================================
start = time.time()
runpy.run_path(path_name=code_path + '\\vertical_fm_FF.py')
end = time.time()

print(80 * '=')
print(f"\n\t Fitting of federated forest on vertical data: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')



# 10. Create plots
#=======================
start = time.time()
runpy.run_path(path_name=code_path + '\\train_plots_inputs.py')
end = time.time()

print(80 * '=')
print(f"\n\t Creating plots showing the learning process: Done! \n\t\t Execution Time: {(end-start)/60:.2f} min. \n")
print(80 * '=')


end_overall = time.time()
print(f"\n\t -> Total execution time: {(end_overall-start_overall)/60:.2f} min.")

