import os
import glob
from datetime import datetime

import pandas as pd
import tqdm
from matplotlib import pyplot as plt

ds = datetime.now().strftime("%Y%m%d_%H%M%S")

trains_ins = []
logs = glob.glob('*ORB.log')
for fp in tqdm.tqdm(glob.glob('*.log')):
    lns = open(fp, 'r').readlines()
    # seek:start,train,eval,end,

    aa = []
    for l in lns:
        # print(l)
        if 'edge' in l and 'mse' in l:
            cur = {}
            # cur['trains'] = []
            cur['mse'] = l[l.find('mse'):l.find('dataset')]
            cur['edge'] = l[l.find('edge'):l.find('mse')]
            aa.append(cur)
        # elif 'Avg Loss:' in l:
        #     cur['trains'].append(l)
        # elif 'Validation Loss:' in l:
        #     cur['vals'].append(l)
        # elif 'Task done' in l:
        #     cur['end'] = l
    trains_ins.append(aa)
print("trains len %s" % len(trains_ins))
# 过滤有结尾的日志，去除训练中断的
full_trains = [i for i in trains_ins if len(i) > 444]

stat_list = []
for train in full_trains:
    # name_model = train['start'][train['start'].find('fingerprints.'):train['start'].find('dataset')]
    # name_dataset = train['start'][train['start'].find('dataset_train_liqiang_'):train['start'].find('\n')]
    iters = []
    for i in train:
        v = i['mse'].replace('mse[', '').replace(']', '')
        mark = i['edge']
        # c = float(i[ps + 9:pe])
        iters.append(float(v))

    # vals = []
    # for i in train['vals']:
    #     ps = i.find('Validation Loss:')
    #     pe = i.find(', Validation Regress')
    #     c = float(i[ps + 16:pe])
    #     vals.append(c)
    # # vals = [i['start'][i['start'].find('Validation Loss:'):i['Validation Regress'].find('\n')] for i in train['trains']]
    # # plt.plot(iters), plt.title("train_iters_%s" % name_model), plt.show()
    # # plt.plot(vals), plt.title("evla_epoch_%s" % name_model), plt.show()
    # cur_model_name = name_model
    # cur_corner_pattern = train['start'][train['start'].find('corner '):]
    # cur_min_loss_eval = vals[-1]
    # cur_min_loss_train = iters[-1]
    stat_list.append({mark: iters})

    # plt.figure(figsize=(18, 12))
    # plt.subplot(2, 1, 1)
    # plt.plot(iters), plt.title("train_iters_%s_%s" % (name_model, name_dataset)),
    # plt.subplot(2, 1, 2)
    # plt.plot(vals), plt.title("evla_epoch_%s_%s" % (name_model, name_dataset))
    # vv = name_dataset.replace('/', '')
    # plt.savefig("J_%s_%s.png" % (00, name_dataset.replace('/','')))
    # plt.show()
pd.DataFrame(stat_list).to_csv("stat_ORB_%s.csv" % ds)
