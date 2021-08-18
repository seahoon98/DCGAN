import torch
from torch import nn

class LogisticRegression(nn.Module): # define the model as a class
    def __init__(self, i): # determine number of input layer for logistic regression model
        super().__init__()
        self.log_reg = nn.Sequential( # sequential model 
                nn.Linear(i, 1), # using Linear layer 'class of cat or not cat'
                nn.Sigmoid() # Sigmoid as an activation function
                )
    def forward(self, x): # forward computation of the model with inputs x
        return self.log_reg(x)

# training models in PyTorch

model = LogisticRegression(16) # initialization of the model. Here LogisticRegression class has 16 input variables

# criterion from which the model learn 
# define cost function for the model 
criterion = nn.BCELoss()

# choose optimizer such as stochastic gradient descent 
# path in model parameters 
# specify different hyper parameters for these optimizer for example learning rate 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epochs = 100
x = torch.rand(16)
y = torch.rand(1)
# train model in number of different epochs

for t in range(n_epochs):
    
    y_pred = model(x)
    # print(y_pred.shape, y.shape, x.shape)

    loss = criterion(y_pred, y) # forward propagation using his criterion binary cross entropy 
    if t == 0:
        print("y_pred\t y\t loss")
    print(y_pred, y, loss)

    # take a step towards optimum
    optimizer.zero_grad()
    loss.backward() # this line ensures a back propagation step 
    optimizer.step() # using stochastic gradient descent on the parameter of small learning rate 

