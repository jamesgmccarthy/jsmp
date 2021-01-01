import datatable as dt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class FinData(Dataset):
    def __init__(self, data, target, date, mode='train', transform=None, cache_dir=None):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.date = date

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.to_list()
        if self.transform:
            return self.transform(self.data.iloc[index].values)
        else:
            if type(index) is list:
                sample = {
                    'target': torch.Tensor(self.target.iloc[index].values),
                    'data':   torch.FloatTensor(self.data[index]),
                    'date':   torch.Tensor(self.date.iloc[index].values)
                }
                return sample
            else:
                sample = {
                    'target': torch.Tensor(self.target.iloc[index]),
                    'data':   torch.FloatTensor(self.data[index]),
                    'date':   torch.Tensor(self.date.iloc[index])
                }
                return sample

    def __len__(self):
        return len(self.data)


def load_data(root_dir, mode, overide=None):
    if overide:
        data = dt.fread(overide).to_pandas()
    elif mode == 'train':
        data = dt.fread(root_dir + 'train.csv').to_pandas()
    elif mode == 'test':
        data = dt.fread(root_dir + 'example_test.csv').to_pandas()
    elif mode == 'sub':
        data = dt.fread(root_dir + 'example_sample_submission.csv').to_pandas()
    return data


def preprocess_data(data, scale=False):
    # data = data.query('weight > 0').reset_index(drop=True)
    data['action'] = ((data['resp'].values) > 0).astype('float32')
    features = [
                   col for col in data.columns if 'feature' in col and col != 'feature_0'] + ['weight']
    for col in features:
        data[col].fillna(data[col].mean(), inplace=True)
    target = data['action']
    date = data['date']
    data = data[features]
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return data, target, features, date
