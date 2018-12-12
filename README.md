## Performance

| Encoder | Space | Sentence length | epochs | Training accuracy | note |
| --- | --- | --- | --- | --- | --- |
| Cohesion tokenizer + L2 Logistic | x | . | . | 0.8933353044638525 | . |
| HangleCNNEncoder | x | 120 | 500 | 0.7577480591945658 | window=\[2, 3, 4, 5\], num filters= 100, max pooling |
| HangleCNNEncoder | o | 140 | 1000 | 0.9002077268316351 | window=\[2, 3, 4, 5\], num filters= 100, max pooling |


## Requires

- bokeh >= 0.12.13
- numpy >= 1.14.0
- soynlp >= 0.0.46
- PyTorch >= 0.4.0