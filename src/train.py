import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import after_subplot

# Define a function to train the model for one epoch.
def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    # If a GPU is available, move the model to GPU.
    if torch.cuda.is_available():
        model.cuda()

    # Set the model to training mode.
    model.train()
    
    train_loss = 0.0

    # Iterate over the training data.
    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # If a GPU is available, move the data and target to GPU.
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # Clear the gradients of all optimized variables.
        optimizer.zero_grad()
        
        # Forward pass: compute predicted outputs by passing inputs to the model.
        output  = model(data)
        
        # Calculate the loss.
        loss_value  = loss(output, target)
        
        # Backward pass: compute gradient of the loss with respect to model parameters.
        loss_value.backward()
        
        # Perform a single optimization step (parameter update).
        optimizer.step()

        # Update average training loss.
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss

# Define a function to validate the model after each epoch.
def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # Set the model to evaluation mode.
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        
        # Iterate over the validation data.
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # If a GPU is available, move the data and target to GPU.
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model.
            output  = model(data)
            
            # Calculate the loss.
            loss_value  = loss(output,target)

            # Update average validation loss.
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)]) # Initialize live loss plot
    else:
        liveloss = None

    valid_loss_min = None # Initialize minimum validation loss
    logs = {} # Initialize logs dictionary

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',threshold=0.01) # Initialize learning rate scheduler

    for epoch in range(1, n_epochs + 1): # Loop over each epoch

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        ) # Compute training loss

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss) # Compute validation loss

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            torch.save(model.state_dict(), save_path) # Save the weights to save_path

            valid_loss_min = valid_loss # Update minimum validation loss

        scheduler.step(valid_loss) # Update learning rate

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    test_loss = 0. # Initialize test loss
    correct = 0.   # Initialize correct predictions count
    total = 0.     # Initialize total predictions count

    with torch.no_grad(): # Disable gradient computation

        if torch.cuda.is_available():
            model = model.cuda() # Move model to GPU if available

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda() # Move data to GPU if available

            logits  = model(data) # Forward pass: compute predicted outputs by passing inputs to the model
            loss_value  = loss(logits , target) # Calculate the loss

            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss)) # Update average test loss

            pred  = logits.data.max(1, keepdim=True)[1] # Convert logits to predicted class

            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu()) # Compare predictions to true label and update correct count
            total += data.size(0) # Update total count

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total)) # Print test accuracy

    return test_loss

    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
