from bokeh.plotting import figure

def lineup(str_a, str_b, scores):
    scores_ = np.zeros(len(str_a))
    j = 0
    for i, a in enumerate(str_a):
        if str_b[j] == a:
            scores_[i] = scores[j]
            j += 1
    return scores_

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