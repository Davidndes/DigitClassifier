import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#### Gathering the data 

training_dataset = torchvision.datasets.MNIST(root='./ML_Project_(Handwriting)', train = True, transform = transforms.ToTensor(), download = True )
testing_dataset = torchvision.datasets.MNIST(root='./ML_Project_(Handwriting)', train = False, transform = transforms.ToTensor(), download = True )

# Checking what's been downloaded, number of samples, size of each sample and corresponding label etc 
print ("Size of training dataset:", len(training_dataset))
print ("Size of testing dataset:", len(testing_dataset))

#translate the images into tensors (a stack of matrices)
trainingBatches = DataLoader(training_dataset, batch_size = 32, shuffle = True)
testBatches = DataLoader(testing_dataset, batch_size = 32, shuffle = True)

### Model Function
class digitRecogition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.firstLayerParam = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3,  stride =1, padding = 1) # Create a convolution layer containing kernels (filters) that will be used to extract features from the images
        self.ReLU = torch.nn.ReLU() # Introduce nonlinearity (ReLU makes all the inputs floored at 0, gives the model more ways to combine / relate weigths mathematically)
        self.firstPooling = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.secondLayerParam = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,  stride =1, padding = 1) # Create a second convolution layer containing kernels (filters) that will be used to extract features from the feature maps produced in the first layer
        self.secondPooling = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.flatLayer = torch.nn.Flatten(start_dim = 1)
        self.Classifier = torch.nn.Linear(in_features = 392, out_features =10) #create the classifer layer thats going to detrmeine what goe into the loss fucntion. This also separtes into 10 isticnt neurons 

    def forward(self, images):
        netWork = self.firstPooling(self.ReLU(self.firstLayerParam(images)))
        netWork = self.secondPooling(self.ReLU(self.secondLayerParam(netWork)))
        netWork = self.flatLayer(netWork)
        netWork = self.Classifier(netWork)
        return netWork 


# Calls a chosen optimizer to use on the data above 
def chooseOptimizer(model, optimizerName, learningRate):

        if optimizerName == ("SGD"):
            return torch.optim.SGD(model.parameters(), lr = learningRate)
        elif optimizerName == ("Adam"):
            return torch.optim.Adam(model.parameters(), lr = learningRate)

model = digitRecogition() #calling the class on a model
lossFunction = torch.nn.CrossEntropyLoss()
optimizer = chooseOptimizer(model, "SGD", 0.1)

            
def trainingSequence(model, trainingBatches, optimizer, lossFunction):
    for ePoch in range(1000):
        
        for images,labels in trainingBatches: #creates an iterator over the list of batches for selecting one bacth at a time     
            #before = model.firstLayerParam.weight.detach().clone()
            optimizer.zero_grad()
            ForwardPass = model(images)
            modelLoss = lossFunction(ForwardPass,labels) # calculate the loss from the forward pass
            modelLoss.backward() # run the backpropogation on the loss
            optimizer.step()
            #after = model.firstLayerParam.weight.detach().clone()
            #print ((after - before).abs().max().item())  
        return model
         
trainedModel = trainingSequence (model, trainingBatches, optimizer, lossFunction) 

trainedModel.eval()

correctPredictions = 0
totalPredictions = 0

for images,labels in testBatches:
        
    with torch.no_grad():
        results = trainedModel(images) 
    Prediction = torch.argmax(results, dim = 1)
    correctPredictions += (Prediction == labels).sum().item()
    totalPredictions += len(labels)

PercentAccurate = (correctPredictions / totalPredictions)

print(PercentAccurate)
print(f"{PercentAccurate:.1%}")






