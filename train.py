import argparse
import torch
import utilities
import fmodel

parser = argparse.ArgumentParser(description='Train the Image Classifier Network CLI App')

# Basic usage: python train.py data_directory
# Options: 
#     * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
#     * Choose architecture: python train.py data_dir --arch "vgg13" 
#     * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
#     * Use GPU for training: python train.py data_dir --gpu

parser.add_argument('data_dir', type=str)
parser.add_argument('--save_dir', type=str, default="./checkpoint.pth")
parser.add_argument('--arch', type=str, default="vgg16")
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--dropouts', type=float)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()
data_directory = args.data_dir
checkpoint_save_path = args.save_dir
lr = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
use_gpu = args.gpu
epochs = args.epochs

def main():
    # Prepare for training
    train_loader, valid_loader, test_loader, train_data = utilities.load_data(data_directory)
    model, criterion, optimizer = fmodel.setup_network(architecture, hidden_units, lr, use_gpu)
    
    # Train the model
    print_every = 5
    steps = 0
    running_loss = 0
    
    print('*********    Training started!    *********')

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            
            if torch.cuda.is_available() and use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            
            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calcualte the accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch: {epoch+1}/{epochs} \t"
                    f"Loss: {running_loss/print_every:.3f} \t"
                    f"Validation Loss: {valid_loss/len(valid_loader):.3f}\t"
                    f"Accuracy: {accuracy/len(valid_loader):.3f}"
                    )
                
                running_loss = 0
                model.train()
                
    
    print('*********    Training ended!    *********')
    
    # TODO: Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    torch.save({'input_size': 25088,
                'output_size': 102,
                'structure': architecture,
                'learning_rate': lr,
                'classifier': model.classifier,
                'epochs': epochs,
                'hidden_units': hidden_units,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, checkpoint_save_path)
    
    print(f"Checkpoint saved to {checkpoint_save_path}")
    
if __name__== "__main__":
    main()