[![Build status](https://github.com/AntoinePassemiers/PORTIA/actions/workflows/build.yml/badge.svg)](https://github.com/AntoinePassemiers/PORTIA/actions?query=build)
[![Code analysis](https://github.com/AntoinePassemiers/PORTIA/actions/workflows/analysis.yml/badge.svg)](https://github.com/AntoinePassemiers/PORTIA/actions?query=analysis)

# PORTIA

<img align="left" src="docs/imgs/portia.svg" />

Lightning-fast Gene Regulatory Network (GRN) inference tool.
This repository also hosts our graph-theoretical Normalised Discounted Cumulative Gain (gtNDCG) score metric for evaluating inferred GRNs. Usage of both PORTIA and gtNDCG is explained below.

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

PORTIA and gtNDCG can be run either:
- From Python, using the library directly
- As standalone scripts

#### Using the library

In Python, create an empty dataset:

```python
import portia as pt

dataset = pt.GeneExpressionDataset()
```

Gene expression measurements can be added with the `GeneExpressionDataset.add` method. `data` must be an iterable (list, NumPy array, etc) of length `n_genes` containing floating point numbers.

```python
exp_id = 1
data = [0, 0, ..., 1.03424, 1.28009]
dataset.add(pt.Experiment(exp_id, data))
```

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

The mode of regulation (sign of regulatory link) can be retrieved by passing the `return_sign` argument. When set to True, both inferred network and sign matrix will be returned. Sign matrix `S` is a matrix of same shape as `M_bar`, where 1 stands for activition, -1 stands for inhibition, and 0 stands for no (self-)regulation.

```python
M_bar, S = pt.run(dataset, tf_idx=tf_idx, method='fast', return_sign=True)
```

Finally, rank and store the results in a text file. `gene_names` is the list of your genes, provided in the correct order.

```python
with open('your_destination/results.txt', 'w') as f:
    for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names, limit=10000):
        f.write(f'{gene_a}\t{gene_b}\t{score}\n')
```

Scoring of the inferred GRN using our gtNDCG metric is done as follows:
```python
tf_mask = np.zeros(n_genes, dtype=bool)
tf_mask[tf_idx] = True
res = graph_theoretic_evaluation(tmp_filepath, G_target, G_pred, tf_mask=tf_mask)
```

where `tmp_filepath` is the name of the temporary file where to store accessibility matrices, in case the same goldstandard network is used multiple times in a row (e.g. to compare GRN inference methods). If `None` is provided, no temporary file will be written. `G_pred` and `G_pred` are NumPy matrices. NaN elements correspond to missing values. For the goldstandard network, a missing value means that there is no experimental evidence for a given gene pair (even for the absence of regulation). For the inferred network, a missing value means the absence of prediction. For `G_target`, 1 corresponds to a regulatory relationship and 0 the absence of such relation. Scores in `G_pred` are real-valued.

#### Run standalone scripts (command line)

`test-data` folder contains in silico-generated data meant for testing PORTIA and the gtNDCG metric scoring algorithm. The following command line infers a GRN from a gene expression dataset, and stores it in `test-data/out1.txt`:
```
python3 run.py test-data/dataset1.expression.txt --out test-data/out1.txt
```

A list of putative TFs and knock-out experiments can be pointed out in separate files:
```
python3 run.py test-data/dataset2.expression.txt --kos test-data/dataset2.kos.txt --tfs test-data/dataset2.tfs.txt --out test-data/out2.txt
```

Shrinkage parameters can be specified with the arguments `--lambda1 0.8` and `--lambda2 0.05`. Providing the `--signed` argument will make the predictions signed, and will thus contain negative values. For more information on the other arguments, you can access the help by running the `run.py` script without argument.

Scoring the inferred network with the gtNDCG metric requires a goldstandard network:

```
python3 run_gt_ndcg.py test-data/out1.txt test-data/dataset1.goldstandard.txt --out test-data/results1
```

Results will be placed in folder `test-data/results1`.

When a list of TFs is available, it should be provided to the script as well:
```
python3 run_gt_ndcg.py test-data/out2.txt test-data/dataset2.goldstandard.txt --tfs test-data/dataset2.tfs.txt --out test-data/results1
```
