#
import os
import random
from config_file import Config


config = Config()

filename_list = config.filename_list
data_path = config.dataset_path

def filereader(filename, dev_num):
    list_train = []
    list_test = []
    file_train = open(data_path + filename + '.task.train', 'r', encoding='gb18030', errors='ignore')
    file_test = open(data_path + filename + '.task.test', 'r', encoding='gb18030', errors='ignore')
    # name = filename.split('_')
    save_dir = data_path + '/new/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file_dev = open(os.path.join(save_dir, filename +'_dev'),'w')
    save_file_trn = open(os.path.join(save_dir, filename + '_trn'), 'w')
    save_file_tst = open(os.path.join(save_dir, filename + '_tst'), 'w')

    all_leng = 0
    for line in file_train:
        label = line[0]
        text = line[2:].replace('\n','')
        list_train.append(text+' ||| '+label)
    # print('OK,begin to write train data !')

    for line in file_test:
        label = line[0]
        text = line[2:].replace('\n','')
        list_test.append(text+' ||| '+label)

    if len(list_train) < 1600:
        theta = 1600 - len(list_train)
        for i in range(theta):
            list_train.append(list_train[i])

    print(len(list_train))

    random.shuffle(list_train)

    for i in range(len(list_train)):
        if i < dev_num:
            all_leng += len(list_train[i])
            save_file_dev.write(list_train[i] + '\n')
        elif i < 1600:
            all_leng += len(list_train[i])
            save_file_trn.write(list_train[i] + '\n')
        else:
            continue

    for j in range(len(list_test)):
        if j < 400:
            all_leng += len(list_test[j])
            save_file_tst.write(list_test[j] + '\n')
    print(filename+' average length is: '+str(all_leng/(2000)))

    save_file_dev.close()
    save_file_trn.close()
    save_file_tst.close()
    file_train.close()
    file_test.close()
    del file_test
    del file_train
    print('****************This work have finished '+filename+' !****************')

for filename in filename_list:
    filereader(filename, 200)
