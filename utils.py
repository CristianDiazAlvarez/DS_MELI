from pandas import Series
import re
import numpy as np
import pandas as pd
# UI resultado en notebook
import ipywidgets as widgets
from IPython.display import display
# Modelo
from sentence_transformers import SentenceTransformer, util
import torch
from torch import Tensor
from typing import List, Union

# Modelo seleccionado para busqueda semantica que incluye entrenamiento en portugues 
# 1 --------------------------------------------------------------
def is_valid_input(period):
    if not isinstance(period, str):
        raise TypeError("period must be a string.")
    if period not in ['d','h', 'm', 's']:
        raise ValueError("period must be: 'd','h', 'm', 's'.")
    return True

def split_delta_components(td: Series,period: str='d'):
    if is_valid_input(period):
        match period:
            case 'd':
                return td.components.days
            case 'h':
                return td.components.hours
            case 'm':
                return td.components.minutes
            case 's':
                return td.components.seconds

# 3 --------------------------------------------------------------
def encode_titles(df: pd.DataFrame, model: SentenceTransformer) -> Union[List[Tensor], Tensor, List[str]]:
    '''
    Funtion that transform to vectors the DataFrame input text using the pretrained model
    Input:
        df: Dataframe con la información de los titulos de los productos
        model: Pre-trained and pre-loaded model to vectorize the text input
    Output:
        pos1: a stacked tensor is returned, with vectorized corpus
        pos2: a list of strings with the titles 
    '''
    corpus = list(df['ITE_ITEM_TITLE'])
    return model.encode(corpus, convert_to_tensor=True), corpus

def get_top_k_similarity(query: str, model: SentenceTransformer,corpus: list, corpus_embeddings: torch.Tensor, top_k:int=5) -> pd.DataFrame:
    '''
    Using pre-trained and pre-loaded model do semantic search over a single corpus previously vectorized.
    Input:
        query: string with the target text to be searched
        model: Pre-trained and pre-loaded model to vectorize the text input
        corpus: a list of strings with the titles, used to get original text
        corpus_embeddings: a stacked tensor is returned, with vectorized corpus
        top_k: number of desire top results
    Output:
        DataFrame with top_k results, based on cosine similarity between query and corpus
    '''
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    base_item = np.repeat(query, top_k)
    most_similar_item = [corpus[x] for x in top_results[1]]
    scores = [x.item() for x in top_results[0]]
    return pd.DataFrame(data=zip(base_item, most_similar_item, scores),
                        columns=['ITE_ITEM_TITLE', 'ITE_ITEM_TITLE', 'Score Similitud (0,1)'])

def dropdown_corpus_eventhandler(corpus_selector):#: widgets, df:pd.DataFrame, output_corpus, model: SentenceTransformer,corpus: list, corpus_embeddings: torch.Tensor, top_k:int=5):
    '''
    Eventhandler setup to display corpus as dropdown widget, when title is selected display dataframe with top_k results of similarity.
    Input:
        corpus_selector: 
        query: string with the target text to be searched
        model: Pre-trained and pre-loaded model to vectorize the text input
        corpus: a list of strings with the titles, used to get original text
        corpus_embeddings: a stacked tensor is returned, with vectorized corpus
        top_k: number of desire top results
    Output:
        DataFrame with top_k results, based on cosine similarity between query and corpus
    '''
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', 600)
    output_corpus.clear_output()
    with output_corpus:
        query = df[df['ITE_ITEM_TITLE'] == corpus_selector.new]['ITE_ITEM_TITLE'].item()

        answer = get_top_k_similarity(query=query,
                         model=model,
                         corpus=corpus,
                         corpus_embeddings=corpus_embeddings,
                         top_k=5)
        display(answer)