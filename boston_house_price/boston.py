import numpy as np
import matplotlib.pyplot as plt

def load_data():
    #导入需要的数据
    datafile = './housing.data'
    data = np.fromfile(datafile,sep=' ')
    #数据形状变形
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0]//feature_num,feature_num])
    #数据归一化
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    maximum,minimum,avgs = training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0)/training_data.shape[0]
    for i in range(feature_num):
        data[:,i] = (data[:,i] - avgs[i])/(maximum[i] - minimum[i])

    training_data = data[:offset]
    test_data = data[offset:]

    return training_data,test_data

class Network(object):
    def __init__(self,sum_of_weight):
        np.random.seed(3)
        self.w = np.random.randn(sum_of_weight,1)
        self.b = 0
    #前向传播
    def forward(self,x):
        z = np.dot(x,self.w) + self.b
        return z
    #均方loss
    def loss(self,z,y):
        error = z - y
        cost = error * error
        cost = np.sum(cost)
        return cost
    #均方梯度
    def gradient(self,x,y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w,axis=0)
        gradient_w = gradient_w[:,np.newaxis]
        gradient_b = z - y
        gradient_b = np.mean(gradient_b)
        return gradient_w,gradient_b
    #更新w,b
    def update(self,gradient_w,gradient_b,eta):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
    #训练模型
    def train(self,training_data,num_epoches=10,batch_size = 100,eta = 0.1):
        losses = list()
        for epo_id in range(num_epoches):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size)]
            for iter_id,mini_batch in enumerate(mini_batches):
                x = mini_batch[:,:-1]
                y = mini_batch[:,-1:]
                z = self.forward(x)
                loss = self.loss(z,y)
                gradient_w,gradient_b = self.gradient(x,y)
                self.update(gradient_w,gradient_b,eta)
                losses.append(loss)
                print('iter:{},loss:{}'.format(iter_id,loss))
        return losses

def main():
    traing_data,test_data = load_data()
    print('traing_data:')
    print(traing_data[:5])
    print('test_data:')
    print(test_data[:5])
    net = Network(13)
    losses = net.train(traing_data,num_epoches=10,batch_size=100,eta=0.1)
    plt.plot(losses)

    z = net.forward(test_data[:,:-1])
    loss = net.loss(z,test_data[:,-1:])
    print('z:\n', z[:5])
    print('y:\n',test_data[:5,-1:])
    print('loss:\n',loss)

if __name__ == '__main__':
    main()