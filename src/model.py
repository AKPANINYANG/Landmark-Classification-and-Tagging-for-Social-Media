import torch
import torch.nn as nn


# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture. The architecture includes several convolutional layers,
        # each followed by a batch normalization layer, a ReLU activation function, and max pooling.
        # After the convolutional layers, the tensor is flattened and passed through several linear layers.
        # Dropout is applied after the first and second linear layers.
        
        self.model = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 224x224x3 -> 224x224x16
            nn.BatchNorm2d(16),  # Batch normalization to maintain the mean activation close to 0 and the activation standard deviation close to 1
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2, 2),  # Max pooling for spatial down-sampling

            # Second convolutional layer
            nn.Conv2d(16, 32, 3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 56x56x32

            # Third convolutional layer
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 28x28x64

            # Fourth convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 14x14x128

            # Fifth convolutional layer
            nn.Conv2d(128, 256, 3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 7x7x256

            # Flatten the tensor output from the convolutional layers
            nn.Flatten(),  

            # Fully connected layer
            nn.Linear(256 * 7 * 7 , 1024),  
            nn.Dropout(p = dropout),   # Dropout for regularization
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # Another fully connected layer
            nn.Linear(1024 , 512),  
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # Output layer with 'num_classes' outputs
            # The final layer has an output size equal to the number of classes.
            # This layer does not include an activation function,
            # as this is included in the loss function used for training.
            nn.Linear(512, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input tensor through each of our operations
        return self.model(x)
    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
