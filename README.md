

1st File:
Creating Folders.py run this file without any input to create all the directories are needed to read and save data
as well as to save results.
__________________________________________________________________________________________

1st File:
Processing_Data.py with two inputs dataset (Arxiv/CNNDM/Pubmed) and task are (train/test/validation)
Get codes from:
	● Preprocessing\processing_Sentences.py
Preprocessing_Data.py: 
	1) Load the initial data from Hugging Face.
	2) Filter the Articles.
	3) Clean the sentences and keep the original ones and merge the short ones.
	4) Save three files in directory :
		●Sentences.json (list of list of sentences)
		●Cleaned_Sentences.json (list of list of sentences)
		●dataset.csv a dataset contains (articles and summaries)
"might shrink the number of sentences here" depend on the other papers

python Preprocessing_Data.py --dataset Arxiv --task train

_________________________________________________________________________________________

2nd File:
Graph_Processing.py Using the functions of Graph_Optimization\Graph_Optimization.py files. 
	● Load the Tokenized Sentences and Tokenized Cleaned Sentences
	● Build the Sentences nodes and  the sentences adjecency edges and optimize them.
	● Save the Tokenized sentences and cleaned sentences (Updating)
	● Save Sentences Nodes as (dataset/(training/testing/Validation)/_Sentences_Nodes.Json
	● Save Sentences Nodes as (dataset/(training/testing/Validation)/_Sentences_Edges.Json

python Graphs_Processing.py --dataset Arxiv --task train
_________________________________________________________________________________________
3rd File:
Ground_truth.py
Compute the ground truth of all sentences in the articles.
_________________________________________________________________________________________
4th File:
Edge_Creating.py Import codes from Edge_processing folder
In this file Computing many Edges with its features using Sentence Transformer model:
	● Compute the features of sentence adj edges.
	● Compute the Sentence Similarity edges with window size 40% and cos_sim score>0.7
	● Determine the included words nodes.
	● Compute WS features and determine the edges.
	● Compute WW features and determine the edges.
python Edges_Creating.py --dataset Arxiv --task train
_________________________________________________________________________________________
5th File:
Graphs_generating.py Import codes from Graph_embeddings.py
In this file get all the saved elements and save it as a graph which will be used later in the model training.
python Graphs_generating.py --dataset Arxiv --task train

__________________________________________________________________________________________
6tg File :
Training