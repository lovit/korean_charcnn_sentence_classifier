from bokeh.plotting import figure
import numpy as np
import re
import torch
import torch.nn.functional as F
from soynlp.hangle import HangleCNNEncoder

hangle_encoder = HangleCNNEncoder()
normalizer = re.compile('[^가-힣0-9]')

def lineup(longer, shorter, scores):
    scores_ = np.zeros(len(longer))
    j = 0
    for i, a in enumerate(longer):
        if shorter[j] == a:
            scores_[i] = scores[j]
            j += 1
    return scores_

def sentiment_score(sent, model, image_size=-1):
    x, sent_ = sent_to_image(sent, image_size)
    scores = _sentiment_score(x, model)
    scores = lineup(sent, sent_, scores)
    return scores, sent_

def sent_to_image(sent, image_size=-1):
    sent_ = normalizer.sub('', sent)
    x = torch.FloatTensor(hangle_encoder.encode(sent_, image_size))
    x = x.resize(1, 1, x.size()[0], x.size()[1])
    return x, sent_

def _sentiment_score(sentence_image, model):
    x = sentence_image
    scores = np.zeros(x.shape[2])

    ranges = [n for n in model.ranges for _ in range(model.convs[0].weight.size()[0])]
    out = [F.relu(conv(x)).squeeze(3) for conv in model.convs]

    # 1 - max pooling for each conv
    out_ = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
    indices = [F.max_pool1d(i, i.size(2), return_indices=True)[1].squeeze(2) for i in out]
    indices = torch.cat(indices, 1).squeeze(0)

    out_ = torch.cat(out_, 1)
    coef = model.fc.weight[1] - model.fc.weight[0]
    influence = (out_ * coef).squeeze(0)

    for b, n, w in zip(indices, ranges, influence):
        b = b.data.numpy()
        e = b + n
        scores[b:e] += w.data.numpy()
    return scores

def draw_score_plot(chars, scores, height=350, width=1000, title=None):
    if title is None:
        title = "Sentiment score"

    idxs = [str(i) for i in range(len(chars))]
    varh = scores.tolist()
    p = figure(x_range=idxs, title=title)
    p.height = height
    p.width = width
    p.xgrid.grid_line_color = None

    p.vbar(x=idxs, top=varh, width=0.9)
    for i, c in enumerate(chars):
        p.text(i+0.3, 0 if varh[i] <= 0 else -0.1 , [c])

    return p