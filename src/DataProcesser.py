
import numpy as np

class DataProcesser:
    def __init__(self, X, y=None, mean_dict=None, missing_value=0):
        self.X = X
        self.y = y
        self.missing_value = missing_value
        self.columns = self.X.columns

        self.mean_dict = mean_dict
        
        self.dataframe = self.X.copy()
        self.dataframe['y'] = self.y
        

    def _input_missing_flags(self):
        missing_columns = []
        for col in self.columns:
            self.dataframe[col+'_miss'] = np.where(self.dataframe[col] == self.missing_value, 1, 0)
            missing_columns.append(col+'_miss')

        self.dataframe['missing_total'] = self.dataframe.loc[:, missing_columns].apply(lambda row: np.sum(row), axis=1)


    def _transform_missing_into_na(self):
        self.dataframe.loc[:, self.columns] = self.dataframe.loc[:, self.columns].replace(to_replace=self.missing_value, value=np.nan)


    def _input_mean(self, train=True, by_class=True):
        if train:
            if (self.y.all() == None) or (by_class == False):
                self.dataframe.loc[:, self.columns] = self.dataframe.loc[:, self.columns].transform(lambda x: x.fillna(x.mean()))
            else:
                self.dataframe.loc[:, self.columns] = self.dataframe.groupby(self.y)[self.columns].transform(lambda x: x.fillna(x.mean()))

        else:
            self.dataframe = self.dataframe.fillna(self.mean_dict)

    def get_means_by_column(self):
        self.mean_dict = {}
        self._transform_missing_into_na()
        
        for col in self.columns:
            self.mean_dict[col] = self.dataframe[col].mean()

        return self.mean_dict


    def process_train_data(self, with_target_column=True):
        self._input_missing_flags()
        self._transform_missing_into_na()
        self._input_mean()

        if with_target_column:
            return self.dataframe
        else:
            return self.dataframe.drop('y', axis=1)


    def process_test_data(self):
        self._input_missing_flags()
        self._transform_missing_into_na()
        self._input_mean(train=False)

        return self.dataframe.drop('y', axis=1)