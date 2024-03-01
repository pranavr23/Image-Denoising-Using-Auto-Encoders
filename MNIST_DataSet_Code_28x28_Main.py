import os
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.optim as optim
import time


data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_path= os.path.join(os.getcwd(),"image_DN/Test")

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([transforms.Grayscale(1),transforms.Resize((28,28)),
transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)

train_dataset.transform = train_transform


m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256


# Use of dataloader to load the dataset
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=False)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


#Setting Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

                                    ###### function Created to show add noise, display plot with images ############

def add_noise(inputs,noise_factor = 0.5):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy   

def show_Original(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    images = batch
    with torch.no_grad():
        print("original image is shown")
        plt.title('Original valid Image')
        time.sleep(0.1)
        plt.imshow(np.reshape(images[0], (28, 28, 1)), interpolation='nearest',cmap='gist_gray')
    plt.show()  
def show_Noisy(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    images = batch            
    with torch.no_grad():
        print("Noisy image is shown")
        plt.title('Noise valid Image')
        time.sleep(0.1)
        plt.imshow(np.reshape(images[0], (28, 28, 1)), interpolation='nearest',cmap='gist_gray')
    plt.show()
def show_test_Original(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    images = batch
    with torch.no_grad():
        print("original image is shown")
        plt.title('Original test Image')
        time.sleep(0.1)
        plt.imshow(np.reshape(images[0], (28, 28, 1)), interpolation='nearest',cmap='gist_gray')
    plt.show()  
    


                                              ######   Creating Class of Encoder and Deconder #######

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    

### Initializing the loss #####

loss_fn = torch.nn.MSELoss()

### Initializing the encoder and decoder object ######
d = 4
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)

### Initializing the  Optimizer ######
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

####### Check if the GPU is available ######
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')


                                         ##### function for training and validation and then plotting ##############


### Training function
def train_model(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.5):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_noisy = add_noise(image_batch,noise_factor)
        image_batch = image_batch.to(device)
        image_noisy = image_noisy.to(device)  

        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer = torch.optim.Adam(params_to_optimize,
                                     lr=1e-4, 
                                     weight_decay=1e-5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
 
    return np.mean(train_loss)
### Validating function
def val_model(encoder, decoder, device, dataloader, loss_fn, noise_factor=0.5):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad(): # No need to track the gradients
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            
            image_noisy = add_noise(image_batch, noise_factor=noise_factor)

            image_noisy = image_noisy.to(device)

            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Calculate loss for this batch
            

            batch_loss = loss_fn(decoded_data, image_batch)
            total_loss += batch_loss
    
    # Calculate average loss across all batches
    avg_loss = total_loss / len(dataloader)
    return avg_loss
#Printing the Original and Noisy Image for train and validation data
def plot_decoded_trainandval(encoder,decoder,dataset,noise_factor=0.5):
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad(): # No need to track the gradients

        for image_batch, _ in loader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch, noise_factor=noise_factor)
            image_noisy = image_noisy.to(device)

            with torch.no_grad():
               
               print("Decoded image is shown")
               rec_img  = decoder(encoder(image_noisy))
               time.sleep(0.05)
               plt.title('train and val Decoded Image')
               plt.imshow(np.reshape(rec_img[0], (28, 28, 1)), interpolation='nearest' ,cmap='gist_gray')
            plt.show()

            break 
        show_Original(image_batch)
        show_Noisy(image_noisy)
                                             #### Running the model for training and validation the model #######

noise_factor = 0.5
num_epochs = 100

# Creating a history_da directory to stote train_loss and val_loss #######
history_da={'train_loss':[],'val_loss':[]}


##### Running the model for training and validation the model :train and evaluate the autoencoder using the functions #######
for epoch in range(num_epochs):
    i = 0
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    ### Training (use the training function)
    train_loss=train_model(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        dataloader=train_loader, 
        loss_fn=loss_fn, 
        optimizer=optim,noise_factor=noise_factor)
    ### Validation  (use the testing function)
    val_loss = val_model(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        dataloader=valid_loader, 
        loss_fn=loss_fn,noise_factor=noise_factor)
    # Print Validationloss
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
# Plot the train and val loss
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    
    
# plotting the training and val data (Original, Noisy and Decoded images)
plot_decoded_trainandval(encoder,decoder,train_data,noise_factor=noise_factor)   
plot_decoded_trainandval(encoder,decoder,val_data,noise_factor=noise_factor)


                                                  ######## Plotting the training and Validation Loss #######

train_loss = history_da['train_loss']
val_loss = history_da['val_loss']

# Get the number of epochs
num_epochs = len(train_loss)

# Create a list of epoch numbers
epochs = range(1, num_epochs + 1)

# Plot the train and val loss
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation Loss')
plt.show()



                                                             ######## Testing the model ############
encoder.eval()
decoder.eval()

# test Function
def test_model(encoder,decoder,dataset,noise_factor=0.5):
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad(): # No need to track the gradients

        for images_batch, _ in loader:
            images_batch = images_batch.to(device)
            image_noisy = add_noise(images_batch, noise_factor=noise_factor)
            latent = encoder(image_noisy)

            rec_img = decoder(latent)
            rec_img = rec_img.cpu()
            plt.title('test Image decoded')
            plt.imshow(np.reshape(rec_img[0], (28, 28, 1)), interpolation='nearest' ,cmap='gist_gray')
            plt.show()
            show_test_Original(images_batch)
            break 


# plotting the training and val data (Original, Noisy and Decoded images)
test_model(encoder,decoder,test_dataset,noise_factor=noise_factor)


    