import torch
import torch.nn as nn
import torch.nn.functional as F

## ------- ResNet ------ ##
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 2, stride=2) # in the case of basic: 2 conv net * num_blocks
        self.layer2 = self._make_layer(block, 32, 2, stride=2) # in the case of basic: 2 conv net * num_blocks
        
        self.fc = nn.Sequential(
            nn.Linear(10240 , 256), # first layer         
            nn.Linear(256 , 64), # second layer
            nn.Linear(64 , 3), # second layer
            nn.Softmax(dim=1),
        )
        
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))        
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  
    

## ------------------------------------------##
## -------- Mammo graph using ResNet ----- ##
class MGClassifier(nn.Module):
    def __init__(self, ):     
        super(MGClassifier,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 15, stride=3, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2), 
            nn.Conv2d(4, 8, 15, stride=2, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  ,
            nn.Conv2d(8, 16, 7, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  ,
        )

        self.net = ResNet(BasicBlock, 3)
        initialize_weights(self)
        
    def forward(self,x_set): 
        x_out = torch.cat((\
                   torch.cat((self.encoder(x_set[:,0,:,:,:]),\
                             self.encoder(x_set[:,1,:,:,:])),\
                            3),\
                  torch.cat((self.encoder(x_set[:,2,:,:,:]), \
                             self.encoder(x_set[:,3,:,:,:])),\
                            3)\
                  ),2)
       # print('merging',x_out.shape)
        y_hat = self.net(x_out)
       # print('res',x_out.shape)
        
        #y_hat = self.cls(torch.cat((out0,out1,out2,out3), 1))
        return y_hat

## ------------------------------------------##

## ------------- Auto encoder -------------------##
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 15, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(8, 16, 15, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 15, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 15, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 15, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        initialize_weights(self)

    def forward(self, x, enc_only = False):
        x = self.encoder(x)
        if enc_only == False:
            x = self.decoder(x)
        return x

## ------------------------------------------##


## ------------- Multi layer perceptron (MLP)--------------##
class MLPAtt (nn.Module):
    def __init__(self, input_size, emb_size = 64, hidden_size = 32, drop_out = 0.3):
        super(MLPAtt, self).__init__()
        self.in_size = input_size
        self.emb_size = emb_size
        self.hid_size = hidden_size
        self.drop_out = drop_out  
        
        self.emb = nn.Sequential(
            nn.Linear(self.in_size , self.emb_size),            
        )

        self.att= nn.Sequential(
            nn.Linear(self.emb_size , self.emb_size),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.emb_size , self.hid_size), # first layer         
            nn.Dropout(drop_out),
            nn.Linear(self.hid_size , 1), # second layer
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, x): 
        emb_x = self.emb(x)
        att_score = self.att(emb_x)
        context_v = torch.mul(emb_x,att_score)    
        output = self.fc(context_v)
        
        return output#, context_v


def initialize_weights(net):
    torch.manual_seed(1)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()