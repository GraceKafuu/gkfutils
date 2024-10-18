import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.utils.data 
import matplotlib as mpl
import matplotlib.pyplot as plt
#设置随机种子，方便复现
torch.manual_seed(0)
torch.cuda.manual_seed(0)



class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet,self).__init__()
        self.conv1=nn.Conv2d(1,32,3,1)
        self.conv2=nn.Conv2d(32,64,3,1)
        self.dropout1=nn.Dropout2d(p=0.3)
        self.dropout2=nn.Dropout2d(p=0.5)
        self.fc1=nn.Linear(9216,128)
        self.fc2=nn.Linear(128,10)
    
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2)
        x=self.dropout1(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout2(x)
        output=self.fc2(x)#没有经过其他的操作（softmax）
        return output
    

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet,self).__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        output=F.relu(self.fc3(x))
        return output
    


def train_teacher(model,device,train_loader,optimizer,epoch):
    model.train()
    trained_samples=0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        
        trained_samples +=len(data)
        progress=math.ceil(batch_idx/len(train_loader)*50)
        print("\rTrain epoch %d:%d/%d,[%-51s]%d%%" % (epoch,trained_samples,len(train_loader.dataset), "-" * progress + '>',progress * 2),end='')


def test_teacher(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss +=F.cross_entropy(output,target,reduction="sum").item()#总结批次损失
            pred=output.argmax(dim=1,keepdim=True)#获取最大对数概率的索引
            correct +=pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /=len(test_loader.dataset)
    
    print("\nTest:average loss:{:.4f},accuracy:{}/{}({:.0f}%)".format(test_loss,correct,len(test_loader.dataset), 100.* correct/ len(test_loader.dataset)))
    return test_loss,correct / len(test_loader.dataset)


def teacher_main():
    epochs=10
    batch_size=64
    torch.manual_seed(0)
    
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
    
    #加载数据集
    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST",train=True,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST",train=False,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=1000,shuffle=True)
    
    model=TeacherNet().to(device)
    optimizer=torch.optim.Adadelta(model.parameters())

    teacher_history=[]

    for epoch in range(1,epochs+1):
        train_teacher(model,device,train_loader,optimizer,epoch)
        loss,acc=test_teacher(model,device,test_loader)
    
        teacher_history.append((loss,acc))
    torch.save(model.state_dict(),"teacher.pt")
    return model,teacher_history


def train_student(model,device,train_loader,optimizer,epoch):
    model.train()
    trained_samples=0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        
        trained_samples +=len(data)
        progress=math.ceil(batch_idx/len(train_loader)*50)
        print("\rTrain epoch %d:%d/%d,[%-51s]%d%%" % (epoch,trained_samples,len(train_loader.dataset), "-" * progress + '>',progress * 2), end='')
        

def test_student(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss +=F.cross_entropy(output,target,reduction="sum").item()#总结批次损失
            pred=output.argmax(dim=1,keepdim=True)#获取最大对数概率的索引
            correct +=pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /=len(test_loader.dataset)
    
    print("\nTest:average loss:{:.4f},accuracy:{}/{}({:.0f}%)".format(test_loss,correct,len(test_loader.dataset), 100.* correct/ len(test_loader.dataset)))
    return test_loss,correct / len(test_loader.dataset)


def student_main():
    epochs=10
    batch_size=64
    torch.manual_seed(0)
    
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
    
    #加载数据集
    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST",train=True,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST",train=False,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=1000,shuffle=True)
    
    model=StudentNet().to(device)
    optimizer=torch.optim.Adadelta(model.parameters())

    student_history=[]

    for epoch in range(1,epochs+1):
        train_student(model,device,train_loader,optimizer,epoch)
        loss,acc=test_student(model,device,test_loader)
        student_history.append((loss,acc))
    torch.save(model.state_dict(),"student.pt")
    return model,student_history


def distillation(student_output, student_loss, teacher_output, temperature=10, alpha=0.25):
    criterion = nn.KLDivLoss(reduction="batchmean")
    disstillation_loss = criterion(F.log_softmax(student_output / temperature, dim=1), F.softmax(teacher_output / temperature, dim=1))
    loss = student_loss * alpha + disstillation_loss * (1 - alpha)
    return loss


def train_student_kd(teacher_model, model,device,train_loader,optimizer,epoch):
    model.train()
    trained_samples=0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        student_loss=F.cross_entropy(output,target)
        teacher_output=teacher_model(data)
        # loss=distillation(output,target,teacher_output,temp=5.0,alpha=0.7)
        loss=distillation(output, student_loss, teacher_output, temperature=10.0, alpha=0.50)
        loss.backward()
        optimizer.step()
        
        trained_samples +=len(data)
        progress = math.ceil(batch_idx/len(train_loader)*50)
        print("\rTrain epoch %d:%d/%d,[%-51s]%d%%" % (epoch,trained_samples,len(train_loader.dataset), "-" * progress + '>',progress * 2),end='')
        

def test_student_kd(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss +=F.cross_entropy(output,target,reduction="sum").item()#总结批次损失
            pred=output.argmax(dim=1,keepdim=True)#获取最大对数概率的索引
            correct +=pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /=len(test_loader.dataset)
    
    print("\nTest:average loss:{:.4f},accuracy:{}/{}({:.0f}%)".format(test_loss,correct,len(test_loader.dataset), 100.* correct/ len(test_loader.dataset)))
    return test_loss,correct / len(test_loader.dataset)


def student_kd_main():
    epochs=10
    batch_size=64
    torch.manual_seed(0)
    
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
    
    #加载数据集
    train_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST",train=True,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        datasets.MNIST("../data/MNIST",train=False,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=1000,shuffle=True)
    
    teacher_model = TeacherNet().to(device)
    teacher_model.load_state_dict(torch.load("teacher.pt"))
    model=StudentNet().to(device)
    optimizer=torch.optim.Adadelta(model.parameters())

    student_kd_history=[]

    for epoch in range(1,epochs+1):
        train_student_kd(teacher_model, model,device,train_loader,optimizer,epoch)
        loss,acc=test_student_kd(model,device,test_loader)
        student_kd_history.append((loss,acc))
    torch.save(model.state_dict(),"student_kd.pt")
    return model,student_kd_history




# def softmax(x):
#     x_exp = np.exp(x)
#     return x_exp/x_exp.sum()


# def softmax_t(x, T):
#     # T是蒸馏温度
#     x_exp = np.exp(x/T)
#     return x_exp/x_exp.sum()


def softmax_t(x, y):
    j = 0
    l = list()
    for i in x:
        X_exp = math.exp(i / y)
        j = j + X_exp
    for i in x:
        l.append(math.exp(i / y)/j )
    return l


if __name__ == '__main__0':
    teacher_model,teacher_history=teacher_main()

    # %matplotlib inline
    epochs=10
    x=list(range(1,epochs+1))

    #测试的精度
    plt.subplot(2,1,1)
    plt.plot(x,[teacher_history[i][1]for i in range(epochs)],label="teacher")

    plt.title("Test accuracy")
    plt.legend()

    #测试的损失
    plt.subplot(2,1,2)
    plt.plot(x,[teacher_history[i][0]for i in range(epochs)],label="teacher")

    plt.title("Test loss")
    plt.legend()


if __name__ == '__main__1':
    student_model,student_history=student_main()

    epochs=10
    x=list(range(1,epochs+1))

    #测试的精度
    plt.subplot(2,1,1)
    plt.plot(x,[student_history[i][1]for i in range(epochs)],label="student")
    plt.title("Test accuracy")
    plt.legend()

    #测试的损失
    plt.subplot(2,1,2)
    plt.plot(x,[student_history[i][0]for i in range(epochs)],label="student")
    plt.title("Test loss")
    plt.legend()



    
if __name__ == '__main__':
    student_kd_model,student_kd_history=student_kd_main()

    epochs=10
    x=list(range(1,epochs+1))

    #测试的精度
    plt.subplot(2,1,1)
    plt.plot(x,[teacher_history[i][1]for i in range(epochs)],label="teacher")
    plt.plot(x,[student_history[i][1]for i in range(epochs)],label="student")
    plt.plot(x,[student_kd_history[i][1]for i in range(epochs)],label="student_kd")

    plt.title("Test accuracy")
    plt.legend()

    #测试的损失
    plt.subplot(2,1,2)
    plt.plot(x,[teacher_history[i][0]for i in range(epochs)],label="teacher")
    plt.plot(x,[student_history[i][0]for i in range(epochs)],label="student")
    plt.plot(x,[student_kd_history[i][0]for i in range(epochs)],label="student_kd")

    plt.title("Test loss")
    plt.legend()


if __name__ == '__main__3':
    test_loader_bs1=torch.utils.data.DataLoader(
    datasets.MNIST("../data/MNIST",train=True,download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,),(0.3081,))
                      ])),
        batch_size=1,shuffle=True)


    teacher_model.eval()
    with torch.no_grad():
        data,target=next(iter(test_loader_bs1))
        data,target=data.to('cuda'),target.to('cuda')
        output=teacher_model(data)

    test_x=data.cpu().numpy()
    y_out=output.cpu().numpy()
    y_out=y_out[0,::]
    print("Output(NO softmax):",y_out)

    plt.subplot(3,1,1)
    plt.imshow(test_x[0,0,::])

    plt.subplot(3,1,2)
    plt.bar(list(range(10)),softmax_t(y_out,1),width=0.3)#画出直方图

    plt.subplot(3,1,3)
    plt.bar(list(range(10)),softmax_t(y_out,10),width=0.3)
    plt.show()

















