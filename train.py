import torchvision
import torchvision.transforms as transforms

# Define the transform (convert to tensor, normalize, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

"""
Inside trainset, there are 50000 sample tuples. Each tuple contains an image and a label.
The image is a 3x32x32 tensor (3 channels, 32x32 pixels).
The label is an integer from 0 to 9.
Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


