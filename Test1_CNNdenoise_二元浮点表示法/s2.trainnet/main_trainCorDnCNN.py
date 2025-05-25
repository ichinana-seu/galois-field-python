import numpy as np
import torch
import torch.nn as nn
import os

from myConvNet_CorDnCNN import CorDnCNN


if __name__ == "__main__":
    # 参数
    device = 'cuda:0'
    epoch_num = 20000              # 200000
    RS_binaryLength = 60
    IO_prefix = 'ReedSolomon_2p4_t3_N15K9_n60k36-AR0.8'            # 数据从“data__前缀”文件夹中读取，向“model__前缀”文件夹中写入
    minibatch_size = 350

    #
    arg_a = 1
    arg_c = 1
    logical_layers = 16
    filter_height = 9
    filter_number = 64


    X_feature_vectorlen = RS_binaryLength
    Y_label_vectorlen = RS_binaryLength
    INPUT_data_directory = f'./data__{IO_prefix}'
    TrainSetX_filename = f"{INPUT_data_directory}/data__{IO_prefix}__trainX_simpleLLR.npybin"
    TrainSetY_filename = f"{INPUT_data_directory}/data__{IO_prefix}__trainY_simpleColorN.npybin"
    TestSetX_filename = f"{INPUT_data_directory}/data__{IO_prefix}__testX_simpleLLR.npybin"
    TestSetY_filename = f"{INPUT_data_directory}/data__{IO_prefix}__testY_simpleColorN.npybin"
    OUTPUT_model_directory = f'./model__{IO_prefix}'
    if not os.path.exists(OUTPUT_model_directory):
        os.makedirs(OUTPUT_model_directory)
    

    # create network
    myNet = CorDnCNN(logical_layers, filter_number, filter_height, X_feature_vectorlen, Y_label_vectorlen).to(device)

    # define loss function
    criterion = myNet.enhancedLoss
    # SGD_Adam
    Train_optimizer = torch.optim.Adam(myNet.parameters(), lr=0.001)
    Train_min_loss = float('inf')

    # 读取数据
    TraindataX = np.fromfile(TrainSetX_filename, dtype=np.float32)
    TraindataX = TraindataX.reshape(-1, X_feature_vectorlen)
    TraindataX_samplenum = np.size(TraindataX, axis=0)
    TraindataY = np.fromfile(TrainSetY_filename, dtype=np.float32)
    TraindataY = TraindataY.reshape(-1, Y_label_vectorlen)
    TraindataY_samplenum = np.size(TraindataY, axis=0)
    assert TraindataX_samplenum == TraindataY_samplenum

    TestdataX = np.fromfile(TestSetX_filename, dtype=np.float32)
    TestdataX = TestdataX.reshape(-1, X_feature_vectorlen)
    TestdataX_torch = torch.from_numpy(TestdataX).to(torch.float32).to(device)
    TestdataX_samplenum = np.size(TestdataX, axis=0)
    TestdataY = np.fromfile(TestSetY_filename, dtype=np.float32)
    TestdataY = TestdataY.reshape(-1, Y_label_vectorlen)
    TestdataY_torch = torch.from_numpy(TestdataY).to(torch.float32).to(device)
    TestdataY_samplenum = np.size(TestdataY, axis=0)
    assert TestdataX_samplenum == TestdataY_samplenum



    epoch_now = 0
    myNet = myNet.to(device)        # to cuda:0
    while epoch_now < epoch_num:
        myNet.train()                       # 设置CNN网络为“训练模式”
        epoch_now += 1
        # 从文件中一个一个读取npybin
        random_index = np.random.choice(TraindataX_samplenum, minibatch_size, replace=False)        # load next minibatch（随机选择minibatch个）
        TrainX_random_torch = torch.from_numpy(TraindataX[random_index]).to(torch.float32).to(device)
        TrainY_random_torch = torch.from_numpy(TraindataY[random_index]).to(torch.float32).to(device)
        # 梯度清零，有pytorch自带记录梯度和运算网络
        Train_optimizer.zero_grad()           # 梯度清零，有pytorch自带记录梯度和运算网络
        Train_outputs = myNet(TrainX_random_torch)
        Train_loss = criterion(Train_outputs, TrainY_random_torch, minibatch_size)
        Train_loss.backward()
        Train_optimizer.step()
        # torch.cuda.empty_cache()

        # 实时评估网络
        if epoch_now % 500 == 0 or epoch_now == epoch_num or epoch_now==1:
            print('Epoch: %d' % epoch_now, end='      ')
            myNet.eval()                        # 设置CNN网络为“实时评估模式”
            # inline function: 
            online_ave_loss_after_train = 0.0
            with torch.no_grad():                       # 什么是original loss
                online_Test_outputs_torch = myNet.forward(TestdataX_torch)
                online_loss_after_training_OBJECT = criterion(online_Test_outputs_torch, TestdataY_torch, TestdataX_samplenum, arg_a, arg_c)
                online_ave_loss_after_train = online_loss_after_training_OBJECT.item()
                # del online_Test_outputs_torch
                # del online_loss_after_training_OBJECT
                # torch.cuda.empty_cache()
                print('OnlineTestLoss: %.4f ' % online_ave_loss_after_train )
            # 临时保存网络
            if online_ave_loss_after_train < Train_min_loss:
                Train_min_loss = online_ave_loss_after_train
                save_model_filename_Temp = f"{OUTPUT_model_directory}/model__{IO_prefix}__simpleTestCorDnCNN_Temp.pth"
                save_model_filename = f"{OUTPUT_model_directory}/model__{IO_prefix}__simpleTestCorDnCNN.pth"
                torch.save(myNet, save_model_filename)           # 临时保存网络
                torch.save(myNet, save_model_filename_Temp)           # 临时保存网络

    save_model_filename_Last = f"{OUTPUT_model_directory}/model__{IO_prefix}__simpleTestCorDnCNN_Last.pth"
    torch.save(myNet, save_model_filename_Last)