def lineup(str_a, str_b, scores):
    scores_ = np.zeros(len(str_a))
    j = 0
    for i, a in enumerate(str_a):
        if str_b[j] == a:
            scores_[i] = scores[j]
            j += 1
    return scores_
