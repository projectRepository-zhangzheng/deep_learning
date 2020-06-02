#导入paddle 需要的包
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random

def load_data():
    #载入需要的数据
    filedata = './housing.data'
    data = np.fromfile(filedata,sep=' ')

    #对数据进行形状变形
    feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_name)
    data = data.reshape([data.shape[0]//feature_num,feature_num])

    #数据归一化
    ratio = 0.8
    offset = int(ratio * data.shape[0])
    training_data = data[:offset]
    test_data = data[offset:]
    maximum, minimum,avgs = training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0)/training_data.shape[0]

    for i in range(feature_num):
        data[:,i] = (data[:,i] - avgs[i]) / (maximum[i] - minimum[i])

    #记录归一化的参数，预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximum
    min_values = minimum
    avg_values = avgs

    training_data = data[:offset]
    test_data = data[offset:]

    return training_data,test_data

class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        #定义一层全连接层，输出维度是1，激活函数为None，既不使用激活函数
        self.fc = Linear(input_dim=13,output_dim=1,act=None)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x

with fluid.dygraph.guard():
    #声明定义好的线性回归模型
    model = Regressor()
    #开启模型的训练模式
    model.train()
    #加载数据
    training_data,test_data = load_data()
    #定义优化算法，这里使用随机梯度下降SGD
    #学习率设置为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01,parameter_list=model.parameters())

with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 10
    BATCH_NUM = 10

    for epoch_id in range(EPOCH_NUM):
        np.random.shuffle(training_data)
        mini_batchs = [training_data[k:k+BATCH_NUM] for k in range(0,len(training_data),BATCH_NUM)]

        for iter_id, mini_batch in enumerate(mini_batchs):

            x = np.array(mini_batch[:,:-1]).astype('float32')
            y = np.array(mini_batch[:,-1:]).astype('float32')

            #讲numpy数据转换为飞浆动态图variable形式
            house_feature = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)

            #前向计算
            predicts = model(house_feature)

            #计算loss
            loss = fluid.layers.square_error_cost(predicts,label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

            #反向传播
            avg_loss.backward()
            #最小化loss，跟新参数
            opt.minimize(avg_loss)

            model.clear_gradients()
    fluid.save_dygraph(model.state_dict(),'LR_MODEL')