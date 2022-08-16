from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
path=glob("log_20220718_114546_*")[0]
errors_train = []
for ln in open(path, 'r').readlines():
    # print(ln)
    if 'Avg Loss' not in ln:
        continue
    errors_train.append(ln[ln.find('Regression Loss'):ln.find(', Avg Class')].replace('Regression Loss', ''))

errors_eval = []
for ln in open(path, 'r').readlines():
    # print(ln)
    if 'Validation' not in ln:
        continue
    errors_eval.append(ln[ln.find('Regression Loss'):ln.find(', Validation Cl')].replace('Regression Loss', ''))
pd.DataFrame({'err_train': errors_train}).to_csv("mb2_log_error_valuesa.csv")
pd.DataFrame({'err_train': errors_train[::18]}).to_csv("mb2_log_error_valuesa2.csv")
pd.DataFrame({'err_eval': errors_eval}).to_csv("mb2_log_error_valuesb.csv")
# plt.plot([float(i.strip()) for i in errors_train])
# plt.xlabel('num. of iters')
# plt.ylabel('mean of errors')
# plt.show()
