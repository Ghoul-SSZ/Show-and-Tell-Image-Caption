from model import Encoder
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as transforms


data_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

cap = dset.CocoCaptions(root='./coco/images/train2014',
                        annFile='./coco/annotations/captions_train2014.json',
                        transform=data_transform)

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(img)
print(target)
#Encoder()