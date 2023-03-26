def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def load_fashion_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms
    dataset = datasets.FashionMNIST(
        '../fashion_mnist_data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def show_image(image, title=''):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
