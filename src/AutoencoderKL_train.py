from functools import partial
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomChoice
from torchvision.transforms.functional import rgb_to_grayscale,rotate
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from torch.optim import Adam
from tqdm import trange
from torch.nn.functional import mse_loss
import torch
import argparse
import logging
import os
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

flip_transform = RandomChoice([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)])
#To do: Full diahedral group


def make_dataset(csv_dict):
    '''
    
    '''
    data_files_train = {"train": csv_dict['train']}
    dataset_train = load_dataset("csv", data_files=data_files_train, split="train", streaming=False)
    data_files_test = {"test": csv_dict['test']}
    dataset_test = load_dataset("csv", data_files=data_files_test, split="test", streaming=False)
    
    return dataset_train, dataset_test

def collate(batch):
    
    """
    Applies random horizontal and vertical flips to a batch images.

    Parameters
    ----------
    batch : list
        A list of dictionaries, each containing a key 'image' and a value of a path to an image file.

    Returns
    -------
    dict
        A dictionary with a key 'images' and a value of a tensor of shape (batch_size, channels, height, width) containing the transformed images.
    """
  
    images = []
    labels = []

    for sample,label in batch:
        image = Image.open(sample['image'])
        image = ToTensor()(image)
        image = flip_transform(image)
        images.append(image)
        labels.append(label)

    image = torch.stack(images)

    return {
        "images" : images
    }



def train_vae(vae_model,data_loaders,model_save_path,log_images_to_path,epochs,learning_rate):
    #training loop based on Binxu Wang's Tutorial on Stable Diffusion Models at ML from Scratch seminar series at Harvard.
    #see code here: https://github.com/Animadversio/DiffusionFromScratch
    logger = logging.getLogger('my_logger')

    #log that training has begun
    logger.info("Training started")

    #set model to train mode
    vae_model.train()

    loss_fn = lambda x,xhat: mse_loss(x, xhat)

    #load optimiser

    optimizer = Adam(vae_model.parameters(), lr=learning_rate)

    #data_loaders
    data_loader_train = data_loaders['train']
    data_loader_test = data_loaders['test']

    tqdm_epoch = trange(epochs)
    steps = 0
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader_train:
            x = x['images']
            x = x.to('cuda')
            x_hat = vae_model(x).sample
            loss = loss_fn(x, x_hat) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            steps+=1
    
        
        logger.info(f'Epoch {epoch} Average Loss: {avg_loss / num_items}')


        # Print the averaged training loss so far.
            # Update the checkpoint after each epoch of training.
        torch.save(vae_model.state_dict(), f'{model_save_path}/ckpt_vae_{epoch}.pth')
        logger.info(f'Check-point saved at {steps}')
        test_image = next(iter(data_loader_test))


        with torch.no_grad():
            output = vae_model(test_image['images'].to('cuda'))
            # create a grid of original and reconstructed images
            images = torch.cat([test_image['images'], output.sample.cpu()], dim=0)
            grid = make_grid(images, nrow=2, normalize=True)
            # plot the grid using matplotlib
            to_pil = ToPILImage()
            grid_pil = to_pil(grid)
            grid_pil.save(f'{log_images_to_path}/grid_{epoch}.png')
            vae_model.train()
    logger.info(f'Trainining completed {steps}')


def main():

    #set-up logger
    logging.basicConfig(filename='KL_train.log', filemode='w', format='%(message)s', level=logging.INFO)
    logger = logging.getLogger('my_logger')

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Train a VAE on Land Cover data with MSE loss')
    # Add arguments
    parser.add_argument('--train_csv', type=str, default='/content/dataset_train.csv', help='Path to the train CSV file')
    parser.add_argument('--test_csv', type=str, default='/content/dataset_test.csv', help='Path to the test CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--save_path', type=str, default='/content', help='Path to save the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--validation_image_out_dir', type=str, default='/content/logs', help='Directory to log images to')
    parser.add_argument('--from_pretrained', type=str, default=None, help='HF directory name to pretrained VAE' )
    # Parse arguments
    args = parser.parse_args()


    #move to GPU
    vae_model = vae_model.to('cuda')

    #load dataset csv path
    train_dataset_csv_path = args.train_csv
    test_dataset_csv_path = args.test_csv

    cvs_dict = {'train':train_dataset_csv_path,
                'test': test_dataset_csv_path
                }
    
    batch_size = args.batch_size
    #get dataset
    dataset_train, dataset_test = make_dataset(cvs_dict) #_ is test
    #load train dataloader
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size,collate_fn=collate,num_workers=2)
    test_dataloader = DataLoader(dataset_test,batch_size=batch_size,collate_fn=collate)

    data_loaders = {'train':train_dataloader, 'test':test_dataloader}

    #epochs
    num_epochs = args.epochs
    #model save path
    save_path = args.save_path
    #learning rate 
    learning_rate = args.lr

    log_images_dir = args.validation_image_out_dir
    #train
    train_vae(vae_model,
              data_loaders = data_loaders,
              epochs = num_epochs,
              model_save_path = save_path,
              log_images_to_path=log_images_dir,
              learning_rate = learning_rate)

if __name__ == '__main__':
    main()