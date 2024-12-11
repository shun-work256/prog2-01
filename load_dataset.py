import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'num of datasets:{len(ds_train)}')

image,target=ds_train[1]
print(type(image),target)

plt.imshow(image,camp='gray_r')
plt.title(target)
plt.show()

image_tensor=transforms.functional.to_image(image)
image_tensor=transforms.functional.to_image(image,dtype=torch.float32,scale=True)
print(image.shape,image.dtype)
print(image.min(),image.dtype())

print(image_tensor.shape,image_tensor.dtype)

for i in range(5):
   for j in range(5):
       k=i*5+j
       image,target=ds_train[k]
       plt.subplot(5,5,k+1)
       plt.imshow(image,camp='gray_r')
       plt.title(target)
plt.show()