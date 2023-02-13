import argparse
from tqdm import tqdm

from utils import load_data
from torch import nn
import torch
from CSHGNN import EGCN
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=400,help="electricity 320")
parser.add_argument("--data_name", help="include electricity,traffic,fg2c2d,mg2c2d,gear2c2d,cr4,mnist,imdb")
parser.add_argument("--err_percent", default=0, help="unreliable label percent")
parser.add_argument("--radius",default=2,help="radius of competence model")

def train(net,train_iter,args,lr,device,window):
    def init_weights(m):
        if  type(m) == EGCN:
            for parameter,(name,params) in zip(m.parameters(),net.named_parameters()):
                if parameter.shape == (256,) or 'bias' in name:
                    continue
                nn.init.xavier_uniform_(parameter)#xavier参数初始化，使每一层的参数均值和方差都保持不变
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on ',device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    loss_list = []
    acc = []
    net.train()
    name = args.data_name
    percent = str(args.err_percent*100)
    time1 = time.time()
    container_x = []
    container_y = []
    # for i,(x,y) in enumerate(train_iter):
    for i, dataset in tqdm(enumerate(train_iter)):
        dataset = torch.tensor(dataset,dtype=torch.float32)
        # print(i, dataset.shape, dataset.dtype)
        if args.data_name == "mnist":
            x = dataset[0].reshape(args.batch_size, -1)
            y = dataset[1].reshape(args.batch_size, )
        else:
            x = dataset[:,:-1]
            y = dataset[:,-1]
        container_x.append(x.to('cuda:0'))
        container_y.append(y.to('cuda:0'))
        if i >= window:
            optimizer.zero_grad()
            y_h = net((container_x,2))
            index = random.sample(range(args.batch_size),int(args.batch_size*percent))
            l = loss(y_h[index],container_y[-1][index].long())
            with torch.no_grad():
                loss_list.append(l.item())
                l_acc = ((args.batch_size - torch.count_nonzero(
                    torch.max(y_h, dim=1)[1] - container_y[-1])) / args.batch_size).cpu()
                acc.append(l_acc.item() )
            l.backward()
            optimizer.step()
            container_x.pop(0)
            container_y.pop(0)
    with open(f'trainloss/{name}_{percent}_{args.radius*100}.txt', 'a') as file:
        file.write(str(loss_list))
        file.write("\n")
        file.write(str(acc))
        file.write("\n")
        file.write("total acc")
        file.write(str(torch.mean(torch.tensor(acc))))
        file.write("\n")
    time2 = time.time()
    print(time2-time1)

def main(args):
    train_iter = load_data(args.data_name, args.batch_size, args.err_percent)
    if args.data_name == "mnist":
        feature_num = 784
        class_num = 10
    elif args.data_name == "imdb":
        feature_num = 10000
        class_num = 2
    elif args.data_name == "electricity":
        feature_num = 30
        class_num = 2
    elif args.data_name == "traffic":
        feature_num = 30
        class_num = 2
    elif args.data_name == "fg2c2d":
        feature_num = 2
        class_num = 2
    elif args.data_name == "mg2c2d":
        feature_num = 2
        class_num = 2
    elif args.data_name == "cr4":
        feature_num = 2
        class_num = 2
    elif args.data_name == "gear2c2d":
        feature_num = 2
        class_num = 2

    net = nn.Sequential(EGCN(num_feature=feature_num, batch_size=args.batch_size, device='cuda:0'),
                        nn.Linear(8, class_num),
                        nn.Softmax(dim=1))
    net.to('cuda:0')
    train(net, train_iter, args, lr=0.002, device='cuda:0', window=3)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
