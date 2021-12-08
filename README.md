[![Build status](https://github.com/AntoinePassemiers/PORTIA/actions/workflows/build.yml/badge.svg)](https://github.com/AntoinePassemiers/PORTIA/actions?query=build)
[![Code analysis](https://github.com/AntoinePassemiers/PORTIA/actions/workflows/analysis.yml/badge.svg)](https://github.com/AntoinePassemiers/PORTIA/actions?query=analysis)

# PORTIA

<img align="left" src="docs/imgs/portia.svg" />

Lightning-fast Gene Regulatory Network (GRN) inference tool.

PORTIA builds on power transforms and covariance matrix inversion to approximate GRNs, and is orders of magnitude faster than other existing tools (as of August 2021).

---

### How to use it

Install the dependencies:

```bash
pip3 -r requirements.txt
```

For using the end-to-end inference algorithm, install dependencies from `requirements-etel.txt` instead.

Install the package:

```bash
python3 setup.py install
```

In Python, create an empty dataset:

```python
import portia as pt

dataset = pt.GeneExpressionDataset()
```

Microarray experiments can be added with the `GeneExpressionDataset.add` method. `data` must be an iterable (list, NumPy array, etc).

```python
for exp_id, data in enumerate(your_data):
    dataset.add(pt.Experiment(exp_id, data))
```

Gene knock-out experiments can be encoded using the `knockout` optional parameter.

```python
dataset.add(pt.Experiment(exp_id, data, knockout=[gene_idx]))
```

where `gene_idx` is the (0-based) index of the gene being knocked out. Dual/multiple knock-out experiments are supported, but won't help in the inference process in any way.

Run PORTIA on your dataset:

```python
M_bar = pt.run(dataset, method='fast')
```

The output `M_bar` is a matrix, where each element `M_bar[i, j]` is a score in the range [0, 1] reflecting the confidence about gene `i` being a regulator for target gene `j`. A whitelist of putative transcription factors can be specified with the `tf_idx` argument. `tf_idx` must be a (0-based) list of gene indices.

```python
M_bar = pt.run(dataset, tf_idx=tf_idx, method='fast')
```

Finally, rank and store the results in a text file. `gene_names` is the list of your genes, provided in the correct order.

```python
with open('your_destination/results.txt', 'w') as f:
    for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names, limit=10000):
        f.write(f'{gene_a}\t{gene_b}\t{score}\n')
```

Real examples on the DREAM datasets are provided in the `scripts/` folder.
