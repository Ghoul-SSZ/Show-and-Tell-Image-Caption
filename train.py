import pickle
import torch.utils.data as data

from model import Encoder, Decoder
from torchvision import models
from vocab import Vocabulary
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import os
import data_func

#params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


learning_rate = 1e-3
epochs = 3
log_step = 10
save_step = 1000
batch_size = 128
num_workers = 2

data_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load vocabulary wrapper
with open('/home/stevenshidizhou/KTH/lab2/coco/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

#cap = dset.CocoCaptions(root='./coco/images/train2014',
#                        annFile='./coco/annotations/captions_train2014.json',
#                        transform=transforms)

data_loader = data_func.get_loader(root='./coco/images/resized2014',
                                   json='./coco/annotations/captions_train2014.json',
                                   vocab=vocab,
                                   transform=data_transform,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers
                                   )
# Encoder & Decoder init
encoder = Encoder().to(device)
decoder = Decoder(vocab_size=len(vocab),
                  embed_size=512,
                  hidden_size=512,
                  num_layers=1).to(device)

# Loss and Optimizer
lossFn = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.batchnorm.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

#Train the model
total_step = len(data_loader)

for epoch in range(epochs):
    for i, (images, captions, lengths) in enumerate(data_loader):
        #lengths = captions

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = lossFn(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()


        # Print log info
        if i % log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
        if (i + 1) % save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(
                'models/', 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            torch.save(encoder.state_dict(), os.path.join(
                'models/', 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
