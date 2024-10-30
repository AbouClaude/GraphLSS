# GraphLSS
Public repository of our paper ["*GraphLSS: Integrating Lexical, Structural, and Semantic Features for Long Document Extractive Summarization*"](https://arxiv.org/abs/2410.21315) 

### How to create GraphLSS document representation? 

A description of each of the Python files required to generate the graph-based document representations used in our long-document summarization benchmarks follows.

-----

#### Preliminaries: 
Run <code>Creating Folders.py</code> without input arguments to create all the directories needed to read and save data and results.

-----

#### Data Preprocessing:
<code>Processing_Data.py</code> requires an input dataset (<code>arXiv</code> or <code>PubMed</code>) and a task, which can be set as <code>train</code>, <code>test</code>, or <code>validation</code>. 

Processing considers: 

1. Loading dataset from Hugging Face.
2. Filter articles (with empty summaries or summaries longer than the source document).
3. Cleaning sentences: sentence tokenization and merge the short sentences with the previous ones.
4. Three files are generated:
   	- <code>Sentences.json</code> (list of list of original sentences)
	- <code>Cleaned_Sentences.json</code> (list of list of processed sentences)
	- <code>dataset.csv</code> (articles and their corresponding summaries)

```
python Preprocessing_Data.py --dataset 'Arxiv' --task 'train'
```
_________________________________________________________________________________________

#### Sentence Processing:
Using the functions of Graph_Optimization\Graph_Optimization.py, <code>Graph_Processing.py</code> will: 

1. Load the tokenized cleaned sentences (<code>Cleaned_Sentences.json</code>)
2. Calculate sentence nodes and sentence order edges, removing duplicates.
3. Re-save the tokenized sentences and cleaned sentences.
4. Save sentence nodes as a JSON file.
5. Save sentence order edges as a JSON file.
```
python Graphs_Processing.py --dataset 'Arxiv' --task 'train'
```
_________________________________________________________________________________________
#### Extractive Labels Calculation:
<code>Ground_truth.py</code> computes the ground truth extractive label for all sentences in the articles contained in the dataset. This is done by maximizing the ROUGE-1 score. 

1. Given a reference summary, initialize optimization by computing ROUGE-1 F1-score for all sentences in a document
2. Sort values from highest to lowest.
3. Set the sentence with the maximum value as the first relevant sentence.
4. Iterate over the sentences and gather the selected relevant sentence/s with the current one
5. Check if the F1-score of the aggregate sentences is higher than the currently selected sentence set.
6. If the current aggregation surpasses the pre-calculated maximum ROUGE-1 score, aggregate to the relevant sentence set and update the maximum value.

_________________________________________________________________________________________
#### Edge Creation:
Using Sentence Transformer, <code>Edges_Creating.py</code> computes sentence-similarity and word-related edges.

1. Load JSON files produced during Sentence Processing.
2. Load Sentence Bert and compute sentence embedding for each sentence.
3. Compute cosine similarity between every pair of sentences in a window size of 40% of the document length.
4. Define an edge if cosine similarity is higher than 0.7, and save the corresponding edges in a JSON file.
5. Considering the preprocessed sentences, lemmatize each word in a sentence and check its Part-Of-Speech (POS).
6. If the POS of the word is an adjective, verb, or noun, define a new word node and calculate tf-idf for word-in-sentence edges.
8. Compute word-word cosine similarity and define new edges.
```
python Edges_Creating.py --dataset 'Arxiv' --task 'train'
```
_________________________________________________________________________________________
#### Generation of GraphLSS Graphs:
Using <code>Graph_embeddings.py</code>, <code>Graphs_generating.py</code> combines all the pre-calculated files and merges saved elements as a PyTorch Geometric graph which will be used later for training a heterogeneous GAT model as a node classification task (sentence nodes).
```
python Graphs_generating.py --dataset 'Arxiv' --task 'train'
```
__________________________________________________________________________________________

