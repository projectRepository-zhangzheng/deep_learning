import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random

def load_data():
    datafile = './housing.data'
    data = np.fromfile(datafile,sep=' ')

    #数据形状变换
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    data = data.reshape([data.shape[0]//feature_num,feature_num])

    #数据归一化
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    maximum,minimum,avgs = training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0)/training_data.shape[0]

    global max_val
    global min_val
    global avg_val

    max_val = maximum
    min_val = minimum
    avg_val = avgs

    for i in range(feature_num):
        data[:,i] = (data[:,i] - avgs[i])/(maximum[i] - minimum[i])

    training_data = data[:offset]
    test_data = data[offset:]

    return training_data,test_data

class Regressor(fluid.dygraph.Layer):
    def __init__(self,name_scope):
        super(Regressor,self).__init__(name_scope)

        name_scope = self.full_name()

        self.fc = Linear(input_dim=13,output_dim=1,act=None)

    def forward(self, inputs):
        z = self.fc(inputs)
        return z

with fluid.dygraph.guard():
    model = Regressor('Regressor')
    model.train()
    training_data,test_data = load_data()
    opt = fluid.optimizer.SGD(learning_rate=0.01,parameter_list=model.parameters())

    epoch_sum = 10
    batch_size = 50
    for epo_id in range(epoch_sum):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size)]
        for iter_id,mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:,:-1]).astype('float32')
            y = np.array(mini_batch[:,-1:]).astype('float32')

            house_feature = dygraph.to_variable(x)
            house_price = dygraph.to_variable(y)

            predict = model(house_feature)

            loss = fluid.layers.square_error_cost(predict,label=house_price)
            avg_loss = fluid.layers.mean(fluid.layers.sqrt(loss))
            if iter_id % 5 ==0:
                print('epoch:{},iter:{},loss is:{}'.format(epo_id,iter_id,avg_loss.numpy()))

            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()