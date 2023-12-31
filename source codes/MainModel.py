# import libraries
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

# Create class for training and testing all models

class MainModel():
    """
    MainModel class for training and testing all models

    Attributes
    ----------
    model : torch.nn.Module
        model to train and test
    dataset_name : str
        name of dataset
    
    name : str
        name of model

    optimizer : torch.optim

    train_loader : torch.utils.data.DataLoader

    validation_loader : torch.utils.data.DataLoader

    test_loader : torch.utils.data.DataLoader
    
    epoch : int
    
         number of epochs        
    """
    def __init__(self,model,dataset_name,name=None)->None:
        self.dataset_name = dataset_name
        self.model_name = name
        self.model=model
        self.optimizer=None
        self.train_loader=None
        self.validation_loader=None
        self.test_loader=None
        self.epoch=0
        self.train_losses=[]
        self.test_losses=[]
        self.train_accuracy=[]
        self.test_accuracy=[]
        self.train_f1_score=[]
        self.test_f1_score=[]
        self.train_precision=[]
        self.test_precision=[]
    
        

    def load_data(self,batch_size=32,train_size=0.8,validation_size=0.2,transform=None,loader_function=None)->None:
        """a function for loading dataset in train,validation and test

        Args:
            batch_size (int, optional): batch size of each epoch. Defaults to 32.
            train_size (float, optional): Train ration of whole dataset. Defaults to 0.8.
            validation_size (float, optional): validation ration  . Defaults to 0.2.
            transform (_type_, optional): . Defaults to None.
            loader_function (_type_, optional): a function for load dataset in specific format  . Defaults to None.
        """
        # check if loader_function is not None
        if loader_function:
            dataset,num_classes = loader_function(self.dataset_name)
        
        # split dataset to train,validation and test
        train_ratio = int(len(dataset)*train_size)
        validation_ratio = int(len(dataset)*validation_size)
        training_set,validation_set,test_set = random_split(dataset,[train_ratio , validation_ratio,len(dataset) - (train_ratio + validation_ratio)])
        self.train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_set,batch_size=batch_size,shuffle=False)
        self.test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

    

     
    def train(self,lr=0.01,weight_decay=0.0005, epochs=30,convert_function=None) -> None:
        """a function for training model 

        Args:
            lr (float, optional):  Learning rate. Defaults to 0.01.
            weight_decay (float, optional) : Weight decay (L2 loss on parameters). Defaults to 0.0005.
            epochs (int, optional): number of epochs. Defaults to 30.
            convert_function (_type_, optional): a fuction for convert data to specific format. Defaults to None.
        """
        self.epoch=epochs
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        size_all_mb = round(size_all_mb,3)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer=optimizer
        for epoch in range(epochs):
            self.model.train()
            for i, data in enumerate(self.train_loader):
                if convert_function:
                    data = convert_function(data)
                optimizer.zero_grad()
                data = data.to("cpu")
                self.model = self.model.to("cpu")
                out = self.model(data)

                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()



            train_acc = self.calculate_accuracy(Train=True,convert_function = convert_function)
            train_loss = self.calculate_loss(Train=True , convert_function = convert_function)
            
            val_acc = self.calculate_accuracy(Train=False,convert_function = convert_function)
            val_loss = self.calculate_loss(Train=False,convert_function = convert_function)
            
            

           
            print("Epoch: {0} | Train Loss: {1} | Train Acc: {2} | Val Loss: {3} | Val Acc: {4}".format(epoch,train_loss,train_acc,val_loss,val_acc,size_all_mb))
            self.save_checkpoint("../checkpoints/graph/{1}_{2}_{0}.pt".format(epoch,self.dataset_name,self.model_name),epoch)
        self.plot_loss(range(epochs),self.train_losses,self.test_losses,save="../results/self-attention-graph-pooling/graph_dataset/{0},{1}_{2}loss.png".format(self.dataset_name,self.model_name,epoch+1))
        self.plot_accuracy(range(epochs),self.train_accuracy,self.test_accuracy,save="../results/self-attention-graph-pooling/graph_dataset/{0},{1}_{2}accuracy.png".format(self.dataset_name,self.model_name,epoch+1))
    
    def get_accuracy(self)-> float:
        """a function for get accuracy of model

        Returns:
            float: accuracy of model in test and train
        """
        test_acc = max(self.test_accuracy)
        train_acc = max(self.train_accuracy)
        # return round with  3 digits
        return round(test_acc*100,3),round(train_acc*100,3)


        
    def save_weights(self,path:str)->None:
        """function for save weights of model

        Args:
            path (str): path for save weights
        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)
    
    def load_weights(self,path:str)->None:
        """function for load weights of model

        Args:
            path (str): path for load weights
        """
        self.model.load_state_dict(torch.load(path))

    def save_checkpoint(self,path:str,epoch:int)->None:
        """function for save checkpoint of model

        Args:
            path (str): path for save checkpoint
            epoch (int): number of epoch
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
    def load_checkpoint(self,path:str)->None:
        """function for load checkpoint of model

        Args:
            path (str): path for load checkpoint
        """


        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
    
    def calculate_accuracy(self,Train=True,convert_function=None)->float:
        """function for calculate accuracy of model

        Args:
            Train (bool, optional): if True calculate accuracy of train else calculate accuracy of test. Defaults to True.
            convert_function (_type_, optional): a fuction for convert data to specific format. Defaults to None.

        Returns:
            float: accuracy of model
        """

        if Train:
            loader=self.train_loader
        else:
            loader=self.validation_loader
        with torch.no_grad():
            self.model.eval()
            correct = 0.
            loss = 0.
            for data in loader:
                if convert_function:
                    data = convert_function(data)
                data = data.to("cpu")
                self.model = self.model.to("cpu")
                out = self.model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
        accuracy=correct / len(loader.dataset)
        if Train:
            self.train_accuracy.append(accuracy)
        else:
            self.test_accuracy.append(accuracy)
        return accuracy
    
    def calculate_loss(self,Train=True,convert_function=None)->float:
        """function for calculate loss of model

        Args:
            Train (bool, optional): if True calculate loss of train else calculate loss of test. Defaults to True.
            convert_function (_type_, optional): a fuction for convert data to specific format. Defaults to None.

        Returns:
            float: loss of model
        """


        if Train:
            loader=self.train_loader
        else:
            loader=self.validation_loader
        with torch.no_grad():
            self.model.eval()
            correct = 0.
            loss = 0.
            for data in loader:
                if convert_function:
                    data = convert_function(data)
                data = data.to("cpu")
                self.model = self.model.to("cpu")
                out = self.model(data)
                loss += F.cross_entropy(out,data.y).item()
        loss=loss / len(loader.dataset)
        if Train:
            self.train_losses.append(loss)
        else:
            self.test_losses.append(loss)
        return loss
    
    def calculate_confusion_matrix(self):
        pass
    
    def calculate_f1_score(self):
        pass
    
    def calculate_precision(self):
        pass
    
    def plot_confusion_matrix(self):
        pass
    
    def plot_loss(self,epochs,train_losses,test_losses,save=None)->None:
        """function for plot loss of model

        Args:
            epochs (int): number of epochs
            train_losses (list): list of train losses
            test_losses (list): list of test losses
            save (str, optional): path for save plot. Defaults to None.

       
        """
        plt.plot(epochs,train_losses,label="train")
        plt.plot(epochs,test_losses,label="test")
        plt.title("Loss Results {0}".format(self.model_name))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        if save:
            plt.savefig(save)
        plt.show()

        
        
    
    def plot_accuracy(self,epochs:int,train_accuracy:list,test_accuracy:list,save:str=None)->None:
        """function for plot accuracy of model

        Args:
            epochs (int): number of epochs
            train_accuracy (list): list of train accuracy
            test_accuracy (list): list of test accuracy
            save (str, optional): path for save plot. Defaults to None.
        """
        plt.plot(epochs,train_accuracy,label="train")
        plt.plot(epochs,test_accuracy,label="test")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("Accuracy Results {0} | Test accuracy:{1}%".format(self.model_name,round(max(test_accuracy)*100,2)))
        plt.legend()
        if save:
            plt.savefig(save)
        plt.show()
    
    def plot_f1_score(self,epochs,train_f1_score,test_f1_score,save=None):
        plt.plot(epochs,train_f1_score,label="train")
        plt.plot(epochs,test_f1_score,label="test")
        plt.xlabel("epochs")
        plt.ylabel("f1_score")
        plt.legend()
        if save:
            plt.savefig(save)
        plt.show()
        
    
    


    

        
        
        
    
    
    

