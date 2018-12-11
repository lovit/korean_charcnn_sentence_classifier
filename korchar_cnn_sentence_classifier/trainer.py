import math
import numpy as np
import torch
import time

from soynlp.hangle import HangleCNNEncoder

def train(model, loss_func, optimizer, texts, label, image_size, epochs, batch_size, use_gpu=False):

    hangle_encoder = HangleCNNEncoder()

    use_cuda = use_gpu and torch.cuda.is_available()
    if use_gpu:
        print('CUDA is {}available'.format('' if use_cuda else 'not '))

    n_data = len(texts)
    n_batch = n_data // batch_size

    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    loss_before = 0.0001

    # Loop over all epochs
    for epoch in range(epochs):

        loss_sum = 0
        t = time.time()

        for i in range(n_batch):

            # select mini-batch data
            b = i * batch_size
            e = min(n_data, (i+1) * batch_size)
            x_batch = np.stack([hangle_encoder.encode(text, image_size=image_size) for text in texts[b:e]])
            y_batch = np.asarray(label[b:e]) # type : numpy.ndarray

            # numpy.ndarray -> torch.Tensor
            x_batch = torch.FloatTensor(x_batch).unsqueeze(1)
            y_batch = torch.LongTensor(y_batch)

            if use_cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Forward -> Backward -> Optimize
            optimizer.zero_grad() # Make the gradient buffer zero
            x_pred = model(x_batch)
            loss = loss_func(x_pred, y_batch)
            loss.backward()
            optimizer.step()

            # cumulate loss
            if use_cuda:
                loss_sum += loss.cpu().data.numpy()
            else:
                loss_sum += loss.data.numpy()

            if i % 10 == 0:
                loss_tmp = loss_sum / (i+1)
                print('\repoch = {}, batch = {} / {}, training loss = {}'.format(
                    epoch, i, n_batch, '%.3f' % loss_tmp), end='', flush=True)

            #break # debug code

        t = time.time() - t
        if epoch % 20 == 0:
            print('\repoch = {}, training loss = {} ({:.2f} sec){}'.format(
                epoch, '%.3f'%(loss_sum / (i+1)), t, ' '*40), flush=True)

        if loss_sum < 0.0001 or (loss_before / loss_sum) >= 1.001:
            print('Early stoped')
            break

    print('\ntraining was done')

    if use_cuda:
        model = model.cpu()

    return model

def predict(model, sents, batch_size=1000, image_size=-1):

    hangle_encoder = HangleCNNEncoder()
    n_data = len(sents)
    n_batch = math.ceil(n_data // batch_size)
    pred_labels = []

    for i in range(n_batch):

        # select mini-batch data
        b = i * batch_size
        e = min(n_data, (i+1) * batch_size)
        x_batch = np.stack(
            [hangle_encoder.encode(text, image_size=image_size)
             for text in texts[b:e]]
        )

        # numpy.ndarray -> torch.Tensor
        x_batch = torch.FloatTensor(x_batch).unsqueeze(1)

        # predict
        y_pred = model(x_batch)
        pred_labels += torch.argmax(y_pred, dim=1).numpy().tolist()
        print('\r%d / %d' % (i, n_batch), end='')
    print()

    return np.asarray(pred_labels)