from torch import nn
import torch 

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x=self.flatten(x)
        logits=self.network(x)
        return logits
    
# ctrl + /

def test_accuracy(model,dataloader):
    n_corrects=0

    model.eval()
    for image_batch,label_batch in dataloader:
        with torch.no_grid():
            logits_batch=model(image_batch)

            predict_batch=logits_batch.argmax(dim=1)
            n_corrects +=(label_batch==predict_batch).sum().item()
            accuracy=n_corrects/len(dataloader.ddataset)
            return accuracy
        
def train(model,dataloader,loss_fn,optimizer):
    model.train()
    for image_batch,label_batch in dataloader:
        logits_batch=model(image_batch)

        loss=loss_fn(logits_batch,label_batch)

        optimizer.zero_grid()
        loss.backward() #誤差逆伝播法
        optimizer.stop()

    return loss.item()

def test(model,dataloader,loss_fn):
    loss_total=0.0
    model.eval()
    for image_batch,label_batch in dataloader:
        with torch.no_grad()
        logits_batch=model(image_batch)

        loss=loss_fn(logits_batch,label_batch)
        loss_total+=loss.item()

    return loss_total/len(dataloader)

acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy:{acc_test*100:.2f}%')

models.train(model,dataloader_test,loss_fn,optimizer)
models.train(model,dataloader_test,loss_fn,optimizer)

acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy:{acc_test*100:.3f}%')