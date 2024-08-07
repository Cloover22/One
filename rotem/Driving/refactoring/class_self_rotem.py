import pandas as pd
import numpy as np

path = 'c:\\Users\\jaeju\\vscode\\Onepredict\\rotem\\Data\\train3_1.csv' # 2023년도 4월 2주차

class Extract_Rotem_srs_Data:
    
    def __init__(self, path_):
        
        # 인스턴스 변수 초기화
        self.path = path_
        self.data = None
        self.dict_data = None
        self.main_df = None

        self.load_data()
        self.preprocessed_feature_type()
        self.preprocessed_df()
        self.get_label()
        self.extract_main_waveform_dataframe()
    
    def load_data(self):
        self.data = pd.read_csv(self.path)
    
    def preprocessed_feature_type(self):
        self.data['dDate'] = pd.to_datetime(self.data['dDate'])
        self.data['VVVF_SDR_TIME'] = pd.to_datetime(self.data['VVVF_SDR_TIME'])
                
        # boolean -> integer
        for col in self.data.columns:
            if self.data[col].dtype == 'bool':
                self.data[col] = self.data[col].astype(int)
    
    def preprocessed_df(self):
        change_to_nonzero = self.data['VVVF_SDR_ATS_SPEED'].ne(0) & self.data['VVVF_SDR_ATS_SPEED'].shift().eq(0)
        index_to_nonzero = self.data.index[change_to_nonzero].tolist()

        change_to_zero = self.data['VVVF_SDR_ATS_SPEED'].eq(0) & self.data['VVVF_SDR_ATS_SPEED'].shift().ne(0)
        index_to_zero = self.data.index[change_to_zero].tolist()
        if index_to_zero:
            del(index_to_zero[0]) # delete index 0
        
        dict_data = {}
        
        min_counts = 60
        max_counts = 200
        
        for idx, (start, end) in enumerate(zip(index_to_nonzero, index_to_zero)):
            end_start = end - start
            if min_counts <= end_start < max_counts:
                dict_data[idx] = self.data[start:end + 1]
        
        self.dict_data = {new_idx: value for new_idx, (old_idx, value) in enumerate(dict_data.items())}
        
    def get_label(self):
        
        def set_label(row):
            if row['VVVF_SDR_PoweringMode'] == 1:
                return 'upper'
            elif row['VVVF_SD_CDR'] == 1:
                return 'lower'
            else:
                return 'constant'
        
        for i in range(len(self.dict_data)):
            self.dict_data[i]['label'] = self.dict_data[i].apply(set_label, axis=1)
    
    def extract_main_waveform_dataframe(self):
        main_waveform_df = {}
        
        for i in range(len(self.dict_data)):
            standard_speed_value = max(self.dict_data[i]['VVVF_SDR_ATS_SPEED']) * 0.8
            standard_line = len(self.dict_data[i][self.dict_data[i]['VVVF_SDR_ATS_SPEED'] > standard_speed_value]) / len(self.dict_data[i])
            
            if standard_line > 0.6:
                main_waveform_df[i] = self.dict_data[i]
                
        self.main_df = {new_idx: value for new_idx, (old_idx, value) in enumerate(main_waveform_df.items())}
    
    def extract_constant_speed_dataframe(self):
        constant_df = pd.DataFrame()
        
        min_speed = 30
        max_speed = 90

        for i in range(len(self.main_df)):
            if min_speed <= max(self.main_df[i]['VVVF_SDR_ATS_SPEED']) < max_speed:
                standard_constant_value = np.mean(self.main_df[i].loc[self.main_df[i]['label'] == 'constant', 'VVVF_SDR_ATS_SPEED'].values) * 0.7
                selected_idx = self.main_df[i][(self.main_df[i]['label'] == 'constant') & (self.main_df[i]['VVVF_SDR_ATS_SPEED'] >= standard_constant_value)].index
                constant_df = pd.concat([constant_df, self.main_df[i].loc[selected_idx]], ignore_index=True)
        
        return constant_df
    
    def extract_upper_speed_dataframe(self):
        upper_df = pd.DataFrame()
        
        for i in range(len(self.main_df)):
            selected_idx = self.main_df[i][self.main_df[i]['label'] == 'upper'].index
            upper_df = pd.concat([upper_df, self.main_df[i].loc[selected_idx]], ignore_index=True)
        
        return upper_df

    def extract_residual_dataframe(self, segment_data, num):
        segment_data['Current_RES'] = segment_data['VVVF_SD_IRMS'] - segment_data['MOTOR_CURRENT']
        segment_data['Voltage_RES'] = segment_data['VVVF_SD_ES'] - segment_data['VVVF_SD_FC']
        segment_data['abs_diff'] = (segment_data['VVVF_SD_IRMS'] - segment_data['MOTOR_CURRENT']).abs()
        
        segment_data = segment_data[segment_data['abs_diff'] >= num]
        segment_data = segment_data[['Current_RES', 'Voltage_RES']]
        
        return segment_data
    
    def extract_speed_range_var_dataframe(self, label_condition):
        current_var_lst = []
        voltage_var_lst = []
        
        for i in range(len(self.main_df)):
            if label_condition == 'constant':
                standard_constant_value = np.mean(self.main_df[i].loc[self.main_df[i]['label'] == label_condition, 'VVVF_SDR_ATS_SPEED'].values) * 0.7
                selected_idx = self.main_df[i][(self.main_df[i]['label'] == label_condition) & (self.main_df[i]['VVVF_SDR_ATS_SPEED'] >= standard_constant_value)].index
                current_var_lst.append(np.var(self.main_df[i].loc[selected_idx, 'VVVF_SD_IRMS'] - self.main_df[i].loc[selected_idx, 'MOTOR_CURRENT']))
                voltage_var_lst.append(np.var(self.main_df[i].loc[selected_idx, 'VVVF_SD_ES'] - self.main_df[i].loc[selected_idx, 'VVVF_SD_FC']))
            elif label_condition == 'upper':
                current_var_lst.append(np.var(self.main_df[i][self.main_df[i].label == label_condition]['VVVF_SD_IRMS'] - self.main_df[i][self.main_df[i].label == label_condition]['MOTOR_CURRENT']))
                voltage_var_lst.append(np.var(self.main_df[i][self.main_df[i].label == label_condition]['VVVF_SD_ES'] - self.main_df[i][self.main_df[i].label == label_condition]['VVVF_SD_FC']))
        
        var_df = pd.DataFrame({'current_var': current_var_lst})
        var_df['voltage_var'] = voltage_var_lst
        
        return var_df

    def extract_constant_var_dataframe(self, label_condition):
        current_var_lst = []
        voltage_var_lst = []
        
        for i in range(len(self.dict_data)):
            current_var_lst.append(np.var(self.dict_data[i][self.dict_data[i].label == label_condition]['VVVF_SD_IRMS'] - self.dict_data[i][self.dict_data[i].label == label_condition]['MOTOR_CURRENT']))
            voltage_var_lst.append(np.var(self.dict_data[i][self.dict_data[i].label == label_condition]['VVVF_SD_ES'] - self.dict_data[i][self.dict_data[i].label == label_condition]['VVVF_SD_FC']))
        
        var_df = pd.DataFrame({'current_var': current_var_lst})
        var_df['voltage_var'] = voltage_var_lst
        
        return var_df

    def extract_constant_std_dataframe(self, label_condition):
        current_std_lst = []
        voltage_std_lst = []
        
        for i in range(len(self.dict_data)):
            current_std_lst.append(np.std(self.dict_data[i][self.dict_data[i].label == label_condition]['VVVF_SD_IRMS'] - self.dict_data[i][self.dict_data[i].label == label_condition]['MOTOR_CURRENT']))
            voltage_std_lst.append(np.std(self.dict_data[i][self.dict_data[i].label == label_condition]['VVVF_SD_ES'] - self.dict_data[i][self.dict_data[i].label == label_condition]['VVVF_SD_FC']))
        
        std_df = pd.DataFrame({'current_std': current_std_lst})
        std_df['voltage_std'] = voltage_std_lst
        
        return std_df

'''
if __name__ == "__main__":

    path = 'c:\\Users\\jaeju\\vscode\\Onepredict\\rotem\\Data\\train3_1.csv' # 2023년도 4월 2주차
    r = Extract_Rotem_srs_Data(path)
    
    constant_df = r.extract_constant_speed_dataframe()
    upper_df = r.extract_upper_speed_dataframe()
    
    res_df = r.extract_residual_dataframe(constant_df, 5)  # editable: constant_df / upper_df
    
    res_df_constant = r.extract_speed_range_var_dataframe('constant')
    res_df_upper = r.extract_speed_range_var_dataframe('upper')
'''