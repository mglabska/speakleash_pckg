# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import pandas as pd
import pickle


from speakleash import Speakleash



PROJECT = input('Enter a dataset name: ')

def get_data(ds):
    lst1 = []
    for doc in ds:
        txt, meta = doc
        meta['text'] = txt
        lst1.append(meta)
    frame = pd.DataFrame(lst1)
    return frame


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(PROJECT))
    replicate_to = os.path.join(base_dir, PROJECT)
    sl = Speakleash(replicate_to)
    ds = sl.get(PROJECT).ext_data
    df = pd.DataFrame(get_data(ds))
    with open(f"datasets/{PROJECT}.pkl","wb") as f:
        pickle.dump(df, f)
    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
