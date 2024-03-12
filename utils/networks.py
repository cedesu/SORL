import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Mnih2015(nn.Module):
    """CNN head similar to one used in Mnih 2015
       (Human-level control through deep reinforcement learning, Mnih 2015)"""
    def __init__(self, image_shape, num_channels, num_actions):
        super(Mnih2015, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(num_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        c_out = self.conv3(self.conv2(self.conv1(torch.randn(1, num_channels, *image_shape))))
        self.conv3_size = np.prod(c_out.shape)
        print("conv3: {}".format(self.conv3_size))

        self.fc1 = nn.Linear(self.conv3_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.conv3_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class Mnih2015_mh(nn.Module):
    """CNN head similar to one used in Mnih 2015
       (Human-level control through deep reinforcement learning, Mnih 2015)"""
    def __init__(self, image_shape, num_channels, num_actions,n_head=3):
        super(Mnih2015_mh, self).__init__()
        self.num_actions = num_actions
        self.n_head=n_head

        self.conv1s = nn.ModuleList([nn.Conv2d(num_channels, 32, 8, stride=4) for _ in range(n_head)])
        self.conv2s = nn.ModuleList([nn.Conv2d(32, 64, 4, stride=2) for _ in range(n_head)])
        self.conv3s = nn.ModuleList([nn.Conv2d(64, 64, 3, stride=1) for _ in range(n_head)])

        c_out = self.conv3s[0](self.conv2s[0](self.conv1s[0](torch.randn(1, num_channels, *image_shape))))
        self.conv3_size = np.prod(c_out.shape)
        print("conv3: {}".format(self.conv3_size))

        self.fc1s = nn.ModuleList([nn.Linear(self.conv3_size, 512) for _ in range(n_head)])
        self.fc2s = nn.ModuleList([nn.Linear(512, num_actions) for _ in range(n_head)])
    
    def forward(self, x,z):
        y_out=0
        for i in range(self.n_head):
            y = F.relu(self.conv1s[i](x))
            y = F.relu(self.conv2s[i](y))
            y = F.relu(self.conv3s[i](y))

            y = y.view(-1, self.conv3_size)
            y = F.relu(self.fc1s[i](y))
            y = self.fc2s[i](y)
            
            y_out+=y*z[i]

        return y_out
    
    def select(self, x,y,mask=None):
        assert mask is None
        y=F.one_hot(y.view(-1).long(),num_classes=self.num_actions)#(bs*l,a)
        y_true=y#*mask.view(-1).unsqueeze(-1)
        prob_list=[]
        
        for i in range(self.n_head):
            y = F.relu(self.conv1s[i](x))
            y = F.relu(self.conv2s[i](y))
            y = F.relu(self.conv3s[i](y))

            y = y.view(-1, self.conv3_size)
            y = F.relu(self.fc1s[i](y))
            y = self.fc2s[i](y)
            
            y_pred=F.softmax(y)#(bs,a)
            prob=(y_true*y_pred).sum()
            
            prob_list.append(prob.unsqueeze(-1))
        prob_list=torch.tensor(prob_list)
        return prob_list/prob_list.sum()#(n_head)
    
class Mnih2015_mh_lora(nn.Module):
    """CNN head similar to one used in Mnih 2015
       (Human-level control through deep reinforcement learning, Mnih 2015)"""
    def __init__(self, image_shape, num_channels, num_actions,n_head=3):
        super(Mnih2015_mh_lora, self).__init__()
        self.num_actions = num_actions
        self.n_head=n_head

        self.conv1 = nn.Conv2d(num_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        c_out = self.conv3(self.conv2(self.conv1(torch.randn(1, num_channels, *image_shape))))
        self.conv3_size = np.prod(c_out.shape)
        print("conv3: {}".format(self.conv3_size))

        self.fc1 = nn.Linear(self.conv3_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self.rank=9#4
        self.lora1_up=nn.ModuleList([nn.Linear(self.conv3_size,self.rank) for _ in range(self.n_head)])
        self.lora1_down=nn.ModuleList([nn.Linear(self.rank,512) for _ in range(self.n_head)])
        self.lora2_up=nn.ModuleList([nn.Linear(512,self.rank) for _ in range(self.n_head)])
        self.lora2_down=nn.ModuleList([nn.Linear(self.rank,num_actions) for _ in range(self.n_head)])
    
    def forward(self, x,z):
        y_out=0
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y_conv = y.view(-1, self.conv3_size)
        for i in range(self.n_head):
            y = F.relu(self.fc1(y_conv)+self.lora1_down[i](self.lora1_up[i](y_conv)))
            y = self.fc2(y)+self.lora2_down[i](self.lora2_up[i](y))
            
            y_out+=y*z[i]

        return y_out
    
    def forward_all_head(self, x,z):
        y_out=[]
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y_conv = y.view(-1, self.conv3_size)
        for i in range(self.n_head):
            y = F.relu(self.fc1(y_conv)+self.lora1_down[i](self.lora1_up[i](y_conv)))
            y = self.fc2(y)+self.lora2_down[i](self.lora2_up[i](y))
            
            y_out.append(y)

        return torch.stack(y_out,dim=0)
    
    def select(self, x,y,mask=None):
        assert mask is None
        y=F.one_hot(y.view(-1).long(),num_classes=self.num_actions)#(bs*l,a)
        y_true=y#*mask.view(-1).unsqueeze(-1)
        prob_list=[]
        
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y_conv = y.view(-1, self.conv3_size)
        for i in range(self.n_head):
            y = F.relu(self.fc1(y_conv)+self.lora1_down[i](self.lora1_up[i](y_conv)))
            y = self.fc2(y)+self.lora2_down[i](self.lora2_up[i](y))
            
            y_pred=F.softmax(y)#(bs,a)
            prob=(y_true*y_pred).sum()
            
            prob_list.append(prob.unsqueeze(-1))
        prob_list=torch.tensor(prob_list)
        return prob_list/prob_list.sum()#(n_head)