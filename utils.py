import numpy as np
import time
import re
import os
from datetime import datetime
import torch
import torch.cuda
import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve, precision_score,recall_score
from scipy import interp   
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from cycler import cycler
import operator

################################################
################## CLASS ##################
################################################
### For early stopping the leaning ######
class EarlyStopping():
    def __init__(self, patience=30, verbose=1, threshold = 0.001):
        self._step = 0
        self._zero = 0
        self._same = 0
        self.threshold = threshold
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose
 
    def validate(self, loss):
        #print(self._loss, loss)
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        elif self._loss <self.threshold: # keep to reach zero 
            self._zero += 1
            if self._zero > self.patience:
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        elif abs(self._loss - loss) < self.threshold:
            self._same += 1
            if self._same > self.patience:
                #print('samesame')
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._same = 0 
            self._zero = 0
            self._loss = loss
 
        return False

## batch dataset
class batch_dataset (Dataset):
    def __init__(self,  x, y):
        self.x_data = torch.FloatTensor(x) 
        self.y_data = torch.FloatTensor(y)
        self.len = len(x)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

#### deep learning model #####
class DeepModel():
    def __init__(self, model,num_classes=2, criterion= torch.nn.CrossEntropyLoss(), \
                lr=0.002, decay=1e-5, \
                note =None):
        self.m = model
        if note ==None: # note: distinguishment for same model with other options
            self.note = datetime.today().strftime("%m%d%H%M_")+str(np.random.randint(10000))
        self.name = model._get_name()+self.note 
        self.criterion =criterion
        self.attention = False
        self.y_pred_prob = None
        self.lr = lr
        self.decay = decay
        self.loss_record = []
        
        ### AUROC
        self.fpr = dict()
        self.tpr = dict()
        self.auroc = dict()
        ### AUPRC
        self.prec = dict()
        self.rec = dict()
        self.auprc = dict()
        
        self.num_classes=num_classes
        
        print('='*50,'\n')
        print('Model name:' ,self.name)
        print('Model Info:', self.m)

            
    def execute_model (self, X_train, y_train, X_test, y_test,  \
                        batch_size=32, epoch =10, stopping_th = 0.001 ,\
                            save = False): 
        
        print('+'*20,'Execution options','+'*20)
        print('- Batch size:',batch_size)
        print('- Num_Epoch:', epoch)        
        self.training(X_train,y_train,batch_size,epoch,stopping_th,)       
        self.test(X_test, y_test,batch_size)
        self.evaluation(models_,y_test,save=save)          

    def test(self, X_test, y_test=None, batch_size=32):
        print('~'*20,'Testing','~'*20) 
        print('Testing data shape:',X_test.shape)  
        t= time.process_time()
        self.m.eval()
        with torch.no_grad():
            for i in range (0,X_test.shape[0],batch_size):
                x = create_variables_GPU(torch.from_numpy(X_test[i:i+batch_size]).float())
                y_hat = self.m.forward(x) 
                if self.y_pred_prob is None:
                    self.y_pred_prob = y_hat.cpu().detach().numpy()
                else:
                    self.y_pred_prob = np.append(self.y_pred_prob, y_hat.cpu().detach().numpy(), axis=0)

            elapsed_time = time.process_time() - t
            self.weight = self.m#.cpu()
            print('***','Complete! Testing time:', '{:.3f}'.format(elapsed_time),'sec ***')
        if y_test is not None:
            self.evaluation(y_test)

    def evaluation(self, y_test, eval_mode = 'micro',save=True):
        print('~'*20,'Evaluation','~'*20) 
        self.fpr = dict()
        self.tpr = dict()
        self.auroc = dict()
        self.prec = dict()
        self.rec = dict()
        self.auprc = dict()
        ## encoding multi-class  to one-hot vector
        if self.num_classes>1:
            y_test = np.eye(self.num_classes)[y_test] 
        # note: for my accommodation, i divided an accuracy computation into two parts such as AUROC and AUPRC  
        ## Understanding micro and macro evaluation in multiclass classification
        # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

        #-----------AUROC---------------#
        ### Micro
        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(y_test.ravel(), self.y_pred_prob.ravel())
        self.auroc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])

        ### Macro
        if self.num_classes>1:  
            for i in range(self.num_classes):
                self.fpr[i], self.tpr[i], _ = roc_curve(y_test[:, i], 
                                                self.y_pred_prob[:, i],pos_label=i)
                self.auroc[i] = auc(self.fpr[i], self.tpr[i])   
            all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.num_classes):
                mean_tpr += interp(all_fpr, self.fpr[i], self.tpr[i])
            mean_tpr /= self.num_classes
            self.fpr["macro"] = all_fpr
            self.tpr["macro"] = mean_tpr        
            self.auroc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])
          

        #-----------AUPRC---------------#
        ### Micro
        self.prec["micro"], self.rec["micro"], _ = precision_recall_curve(y_test.ravel(), self.y_pred_prob.ravel())
        self.auprc["micro"]  = average_precision_score(y_test.ravel(), self.y_pred_prob.ravel(), 'micro')

        ### calculate each class
        if self.num_classes>1:
            for i in range(self.num_classes):
                self.prec[i], self.rec[i], _ = precision_recall_curve(y_test[:,i], self.y_pred_prob[:, i],pos_label=i)
                self.auprc[i]  = average_precision_score(y_test[:, i], self.y_pred_prob[:, i])

        ## save plots#
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        np.savetxt ('./log/' + self.name+'_'+str(self.auroc[eval_mode])+'.roc' ,\
                    np.column_stack((self.fpr[eval_mode],self.tpr[eval_mode])), fmt='%.4f')  
        
        if self.num_classes <=2: # binary case
            eval_mode =1
        np.savetxt ('./log/'+ self.name+'_'+str(self.auprc[eval_mode])+'.prc' ,\
                    np.column_stack((self.rec[eval_mode],self.prec[eval_mode])), fmt='%.4f')
                

    def training(self,X_train,y_train,batch_size=32, epoch =10, stopping_th =0.001,save=True):
        early_stopping = EarlyStopping(threshold = stopping_th)
        deep_train_set = DataLoader(dataset=batch_dataset(X_train,y_train),
                                    batch_size=batch_size, shuffle=True)
        
        print('Training data shape:',X_train.shape, y_train.shape,)        
        print('~'*20,'Training','~'*20)
        self.m = model_on_GPU(self.m)            
        optimizer = torch.optim.Adam(self.m.parameters(), lr=self.lr, weight_decay=self.decay)
        t = time.process_time()      
        for it in range(epoch):
            epoch_loss=0
            epoch_output = None
            for j, (x, y) in enumerate(deep_train_set, 1):
                x_ = create_variables_GPU(x)
                output = self.m.forward(x_)
                if str(self.criterion) == 'CrossEntropyLoss()':
                    y_ = create_variables_GPU(y.long())
                    loss = self.criterion(output, y_) 
                elif str(self.criterion) == 'BCELoss()':
                    y_ = create_variables_GPU(y.float().reshape(-1,1))  
                    loss = self.criterion(output, y_)
                elif str(self.criterion) == 'MSELoss()':
                    y_ = create_variables_GPU(y.float()) 
                    loss = self.criterion(output, y_)
                ## log per batch
                epoch_loss += loss.item() # add a loss value
                if epoch_output is None: # add a sub list of prediction for a batch
                    epoch_output = output.cpu().detach().numpy()
                else:
                    epoch_output = np.append(epoch_output, output.cpu().detach().numpy(), axis=0)
                ## backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ## log per epoch         
            self.loss_record.append(epoch_loss)            
            ## check early stop point
            if early_stopping.validate(epoch_loss): break           
            if(it%10==0):
                print('{:,}th loss: {:.4f}'.format(it, epoch_loss),end=' | ')    
                if str(self.criterion) is not  'MSELoss()':
                    chk_overfitting(epoch_output,y_train)      
            if((it%50==0) & (it>0)& (save ==True)):
                if not os.path.exists('./model/'): os.makedirs('./model/')
                print('Intermediate model saved')
                torch.save(self.m,'./model/'+self.name)
        self.m.cpu()
        print('{:,}th loss:{:.5f}'.format(it, epoch_loss))
        if not os.path.exists('./model/'): os.makedirs('./model/')
        torch.save(self.m,'./model/'+self.name)
        print('*** Complete! Training time:', '{:.3f}'.format(time.process_time() - t),'sec ***')
          
            
#############################################################    
####################### performance evaluation #######################
#############################################################
       
def performance_comparison (models,eval_mode='micro', baseline = 0.1,save=False):
    date = datetime.today().strftime("%m%d%H%M%S")
    print('Performance Evaluation: ',date)
    ## print ranking
    print('AUC Ranking!')
    models.sort(key=operator.attrgetter('auroc'), reverse=True)
    for model in models:
        print(model.name,'{:.3f}'.format(model.auroc[eval_mode]))
    print('*'*30)
    print('Average Precision Ranking!')
    models.sort(key=operator.attrgetter('auprc'), reverse=True)
    for model in models:
        print(model.name,'{:.3f}'.format(model.auprc[eval_mode]))
        
    plt.style.use('dark_background')  ## optional 
    #plt.style.use('seaborn-bright')
    fig_size=(16,8)        
    fig = plt.figure(figsize=fig_size)    
    
    plt.rcParams["axes.prop_cycle"] = cycler('color', 
                                             [  '#CC6677','#DDCC77', '#117733', '#88CCEE', '#6699CC', '#44AA99', '#661100', 
                                              '#882255', '#999933','#AA4499','#AA4466','#332288',])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    ### AUROC ###
    ax1 = fig.add_subplot(1,2,1)
    for model, color in zip(models, colors):    
        ax1.plot(model.fpr[eval_mode], model.tpr[eval_mode], #i 번째 모델에 fpr 과 tpr
                 label=model.name+': {:.3f}'.format(model.auroc[eval_mode]),
                 color=color, linestyle=':', linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k',color='grey', lw=1)
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])

    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_title('Receiver Operating Characteristic', fontsize=14)
    ax1.set_xlabel('1-Specificity', fontsize = 11)
    ax1.set_ylabel('Sensitivity', fontsize = 11)

    ax1.legend(fontsize=12)

    ### AUPRC ###
    if model.num_classes <=2:
        eval_mode =1
    ax2 = fig.add_subplot(1,2,2)
    for model, color in zip(models, colors):    
        ax2.plot(model.rec[eval_mode], model.prec[eval_mode], 
                 label=model.name+': {:.3f}'.format(model.auprc[eval_mode]),
                 color=color, linestyle=':', linewidth=2)

    ax2.plot([0, 1], [baseline, baseline], 'k',color='grey' , lw=1) ## baseline
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])

    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title('Precision Recall Curve', fontsize=14)
    ax2.set_xlabel('Recall', fontsize = 11)
    ax2.set_ylabel('Precision', fontsize = 11)
    ax2.legend(fontsize=11)

    ## save result
    if save ==True:
        save_file_name = date+'.png'
        fig.savefig(save_file_name ,dpi=300, transparent=True)
    plt.show()

        


########################################################################
################################## GPU mode ############################
########################################################################
def create_variables_GPU(tensor):
    if torch.cuda.is_available():
       # print("Variables on Device memory")
        return Variable(tensor.cuda())
    else:
       # print("Variables on Host memory")
        return Variable(tensor)
    
def model_on_GPU (model):
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), 'GPUs used!')
        model = torch.nn.DataParallel(model).cuda()

    elif torch.cuda.is_available():
        print("Model on Device memory")
        model = model.cuda()
    else:
        print("Model on Host memory")
    return model         
########################################################################



    
#### check overfitting #######
def chk_overfitting(pred, true):
    n_classes = np.size(pred,1) # num_variables
    if n_classes>1:
        true = np.eye(n_classes)[true] 
        pred[pred>=((n_classes-1)/n_classes)]=1
        pred[pred!=1]=0
    else:
        pred[pred>=0.5]=1
        pred[pred<0.5]=0
    print('Precision:','{:.4f}'.format(precision_score(true, pred,average='weighted')),end=' | ')
    print('Recall:', '{:.4f}'.format(recall_score(true, pred,average='weighted')))

### oversampling ###
#import  ACGAN_LOS as acgan
#import  GAN_LOS as gan
from imblearn.over_sampling import SMOTE, ADASYN
def over_sampling(X,y,method='SMOTE'): 
    unique, counts = np.unique(y, return_counts=True)
    gen_num = counts[0] - counts[1]
    print('would be generated as many as', gen_num)
    
    if method == 'SMOTE':
        print('Oversampling...SMOTE')
        X_new, y_new = SMOTE(k_neighbors=counts[1]//2, random_state = 123).fit_sample(X, y)
    elif method == 'ADASYN': 
        print('Oversampling...ADASYN')
        X_new, y_new = ADASYN( ratio='minority', n_neighbors=counts[1]//2, random_state = 123).fit_sample(X, y)
    # elif type =='GAN':
    #     #print(len(d_train_x[np.where(d_train_y==1)]))
    #     norm_gan = gan.GAN(X[np.where(y==1)], y[np.where(y==1)],num_z=gen_num, visualize=False)
    #     norm_gan.train()
    #     gen_x, gen_y = norm_gan.generate_samples()
    #     X = np.append(X, gen_x,axis=0)
    #     y = np.append(y, gen_y)
    # elif type =='ACGAN':
    #     ac = acgan.ACGAN(X, y,num_z=gen_num, visualize=True)
    #     ac.train()
    #     gen_x, gen_y = ac.generate_samples()
    #     X = np.append(X, gen_x,axis=0)
    #     y = np.append(y, gen_y)

    print('Oversampled Training dataset',X.shape,'->' ,X_new.shape)   
    return X_new, y_new