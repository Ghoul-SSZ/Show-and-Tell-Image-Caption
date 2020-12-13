from torchvision import models
from torch import nn
import torch
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Encoder: use RESNET pretrained model but
    fine tune it using our training dataset

    For dev purpose, using resnet-34,
    For prod, using resnet-152
    """

    def __init__(self):
        super(Encoder, self).__init__()
        pretrained_model = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(pretrained_model.fc.in_features, out_features=512)
        self.batchnorm = nn.BatchNorm1d(num_features=512, momentum=0.01)

        # init weights randomly with normal distribution
        nn.init.xavier_normal_(self.linear.weight, gain=1)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        """
        Forward propagation.
        @param
            images: images, a tensor of dimensions
        return:
            image embeddings:
                else: (batch_size, embed_size)
        """
        x = self.resnet(x)
        x = Variable(x.data)
        x = x.view(x.size(0), -1)  # flatten
        x = self.linear(x)

        return x


class Decoder(nn.Module):
    """
    Decoder: use RNN with LSTM
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """
        Forward pass through the network
        :param features: features from CNN feature extractor
        :param captions: encoded and padded (target) image captions
        :param lengths: actual lengths of image captions
        :returns: predicted distributions over the vocabulary
        """

        # embed tokens in vector space
        embeddings = self.embed(captions)
        # append image as first input
        #print(features.size()) torch.Size([128, 1000])

        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # pack data (prepare it for pytorch model)
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)

        # run data through recurrent network
        hiddens, _ = self.lstm(packed)
        # return hiddens
        # hiddens here is a PackedSequence, hiddens[0] is the data tensor
        # hiddens[1] is the batch_size tensor
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, max_seg_length=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

