import os
import pandas as pd
import pickle
import plotly.express as px

from speakleash import Speakleash

PROJECT = input('Enter a dataset name: ')


class Chart:
    def __init__(self, PROJECT):
        self.PROJECT = PROJECT
        base_dir = os.path.join(os.path.dirname(self.PROJECT))
        replicate_to = os.path.join(base_dir, self.PROJECT)
        sl = Speakleash(replicate_to)
        self.ds = sl.get(self.PROJECT).ext_data
        self.meta_frame = self.get_meta
        self.df_charts = self.get_data

    @property
    def get_meta(self):
        lst1 = []
        for doc in self.ds:
            txt, meta = doc
            meta['text'] = txt
            lst1.append(meta)
        meta_frame = pd.DataFrame(lst1)
        return meta_frame

    @property
    def get_data(self):
        df = self.meta_frame.copy()
        df = df.drop_duplicates(subset=['text'], ignore_index=True)
        cols = ['punctuations', 'symbols', 'stopwords', 'oovs', 'pos_num', 'pos_x', 'capitalized_words']
        for col in cols:
            df[f'{col}_ratio'] = df[col] / df['words']
        df_charts = df[[
            'avg_sentence_length',
            'avg_word_length',
            'verb_ratio',
            'noun_ratio',
            'punctuations_ratio',
            'symbols_ratio',
            'stopwords_ratio',
            'oovs_ratio',
            'lexical_density',
            'camel_case',
            'capitalized_words_ratio',
            'pos_x_ratio',
            'pos_num_ratio',
            'gunning_fog']]
        return df_charts


    def draw_charts(self):
        for i in self.df_charts.columns:
            fig = px.histogram(self.df_charts[i], title=i, labels=self.df_charts[i].values)
            fig.write_html(f"{self.PROJECT}_{i}_hist.html")


if __name__ == '__main__':
    charts = Chart(PROJECT)
    charts.draw_charts()
