import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(pretrained=True) # Load the pretrained model
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False # Freeze the parameter

    # Add the linear layer at the end with the appropriate number of classes
    num_ftrs  = model_transfer.fc.in_features # Get numbers of features extracted by the backbone

    # Create a new linear layer with the appropriate number of inputs and outputs
    model_transfer.fc  = nn.Sequential(
        nn.Linear(num_ftrs, 1024), # Linear layer with num_ftrs inputs and 1024 outputs
        nn.ReLU(),                 # ReLU activation function
        nn.Dropout(0.5),           # Dropout layer with p=0.5
        nn.Linear(1024, n_classes) # Linear layer with 1024 inputs and n_classes outputs
    )
    
    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
