import pandas as pd
import numpy as np

path = 'c:\\Users\\jaeju\\vscode\\Onepredict\\rotem\\Data\\train3_1.csv' # 2023년도 4월 2주차
path2 = 'c:\\Users\\jaeju\\vscode\\Onepredict\\rotem\\Data\\train3_2.csv'

class Extract_Rotem_srs_Data:
    
    def __init__(self):
        pass
    
    def load_data(self, path_):
        return pd.read_csv(path_)
    
    '''
    preprocessed feature type 
    '''
    def preprocessed_feature_type(self, data):
        
        data['dDate'] = pd.to_datetime(data['dDate'])
        data['VVVF_SDR_TIME'] = pd.to_datetime(data['VVVF_SDR_TIME'])
                
        # boolean -> integer
        for col in data.columns:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype(int)
        
        return data
    
    
    '''
    preprocessed Data Frame 
    '''
    def preprocessed_df(self,data):
        # speed 0 -> ? index
        change_to_nonzero = data['VVVF_SDR_ATS_SPEED'].ne(0) & data['VVVF_SDR_ATS_SPEED'].shift().eq(0)
        index_to_nonzero = data.index[change_to_nonzero].tolist()

        # speed ? -> 0 index
        change_to_zero = data['VVVF_SDR_ATS_SPEED'].eq(0) & data['VVVF_SDR_ATS_SPEED'].shift().ne(0)
        index_to_zero = data.index[change_to_zero].tolist()
        del(index_to_zero[0]) # delete index 0
        
        # 선택된 df 딕셔너리로 관리
        dict_data = {}
        
        # Editable
        '''
        df 길이가 60 ~ 200 인 구간만 추출 (약 60 ~ 200 sec)
        '''
        min_counts = 60
        max_counts = 200
        
        for idx, (start, end) in enumerate(zip(index_to_nonzero, index_to_zero)):
            end_start = end - start
        
            # df 이상치 제거    
            if min_counts <= end_start < max_counts :
                dict_data[idx] = data[start : end +1]
            
        dict_data = {new_idx: value for new_idx, (old_idx, value) in enumerate(dict_data.items())} # 딕셔너리 키 값 재할당
        
        return dict_data
        
    
    def get_label(self, dict_data):
        
        
        def set_label(row):
            '''
            명목 변수 조합 :
            [UPPER 분류] -> 1 로 분류
            VVVF_SDR_ENPOW, VVVF_SDR_PoweringMode, VVVF_SDR_P, VVVF_SD_P

            [Constant 분류] -> 0으로 분류
            VVVF_SD_GATEON, VVVF_BLD_CUR_Valid

            [Lower 분류] -> 1로 분류
            VVVF_SDR_BrakingMode, VVVF_SD_CDR, VVVF_SD_B
            '''
            
            if row['VVVF_SDR_PoweringMode'] == 1:
                return 'upper'                # upper
            elif row['VVVF_SD_CDR'] == 1:
                return 'lower'                # lower
            else:
                return 'constant'                # constant
        
        
        for i in range(len(dict_data)):
            dict_data[i]['label'] = 0
            dict_data[i]['label'] = dict_data[i].apply(set_label, axis=1)
        
        return dict_data
    
    
    def extract_main_waveform_dataframe(self, labeld_data):
        
        main_waveform_df = {}

        # 대표 파형 (ㄷ자) 추출 알고리즘
        for i in range(len(labeld_data)):
            
            standard_speed_value = max(labeld_data[i]['VVVF_SDR_ATS_SPEED'])* 0.8
            standard_line = len(labeld_data[i][labeld_data[i]['VVVF_SDR_ATS_SPEED'] > standard_speed_value]) / len(labeld_data[i])
            
            if standard_line > 0.6 :
                main_waveform_df[i] = labeld_data[i]
                
        main_waveform_df = {new_idx: value for new_idx, (old_idx, value) in enumerate(main_waveform_df.items())}
        
        
        return main_waveform_df
    
    
    def extract_constant_speed_dataframe(self, main_waveform_data):
        
        constant_df = pd.DataFrame()
        
        # Range Editable
        min_speed = 30
        max_speed = 90

        # 정속 구간 추출 알고리즘 
        for i in range(len(main_waveform_data)):
                    
            if min_speed <= max(main_waveform_data[i]['VVVF_SDR_ATS_SPEED']) < max_speed :
                standard_constant_value = np.mean(main_waveform_data[i].loc[main_waveform_data[i]['label'] == 'constant', 'VVVF_SDR_ATS_SPEED'].values) * 0.7
                # 끝 부분 밀리는 데이터 제거
                selected_idx = main_waveform_data[i][(main_waveform_data[i]['label'] == 'constant') & (main_waveform_data[i]['VVVF_SDR_ATS_SPEED'] >= standard_constant_value)].index 
                constant_df= pd.concat([constant_df, main_waveform_data[i].loc[selected_idx]], ignore_index=True)
                            
        return constant_df
    
    
    def extract_upper_speed_dataframe(self, main_waveform_data):
        
        upper_df = pd.DataFrame()
        
        # 가속 구간 추출
        for i in range(len(main_waveform_data)):
            selected_idx = main_waveform_data[i][main_waveform_data[i]['label'] == 'upper'].index
            upper_df= pd.concat([upper_df, main_waveform_data[i].loc[selected_idx]], ignore_index=True)
                            
        return upper_df


    def extract_residual_dataframe(self, 
                                   constant_upper_segment_data,
                                   num):
        
        '''
        잔차 반환 알고리즘
        constant_upper_segment_data : 원하는 가동 구간 데이터,
                                      정속 : extract_constant_speed_dataframe 의 return 값
                                      가속 : extract_upper_speed_dataframe 의 return 값
        num : 원하는 절댓값 수
        '''
        
        constant_upper_segment_data['Current_RES'] = constant_upper_segment_data['VVVF_SD_IRMS'] - constant_upper_segment_data['MOTOR_CURRENT']
        constant_upper_segment_data['Voltage_RES'] = constant_upper_segment_data['VVVF_SD_ES'] - constant_upper_segment_data['VVVF_SD_FC']
        constant_upper_segment_data['abs_diff'] = (constant_upper_segment_data['VVVF_SD_IRMS'] - constant_upper_segment_data['MOTOR_CURRENT']).abs()
        
        constant_upper_segment_data = constant_upper_segment_data[constant_upper_segment_data['abs_diff'] >= num]  # 잔차의 절댓값이 num 보다 큰 데이터만 반환
        constant_upper_segment_data = constant_upper_segment_data[['Current_RES', 'Voltage_RES']] # 잔차만 반환
        
        return constant_upper_segment_data
    
    '''
    정속 / 가속 잔차의 분산 데이터 프레임 추출 
    '''
    
    def extract_speed_range_var_dataframe(self, main_waveform_data, label_condition):
    
        current_constant_var_lst = []
        voltage_constant_var_lst = []
        
        for i in range(len(main_waveform_data)):
            
            # 각 SAMPLE 의 정속 구간의 잔차 분산 추출 
            if label_condition == 'constant' :
                
                standard_constant_value = np.mean(main_waveform_data[i].loc[main_waveform_data[i]['label'] == label_condition, 'VVVF_SDR_ATS_SPEED'].values) * 0.7
                # 끝 부분 밀리는 데이터 제거
                selected_idx = main_waveform_data[i][(main_waveform_data[i]['label'] == label_condition) & (main_waveform_data[i]['VVVF_SDR_ATS_SPEED'] >= standard_constant_value)].index 
                
                current_constant_var_lst.append(np.var(main_waveform_data[i].loc[selected_idx, 'VVVF_SD_IRMS'] - main_waveform_data[i].loc[selected_idx, 'MOTOR_CURRENT']))
                voltage_constant_var_lst.append(np.var(main_waveform_data[i].loc[selected_idx, 'VVVF_SD_ES'] - main_waveform_data[i].loc[selected_idx, 'VVVF_SD_FC']))
                
                var_df = pd.DataFrame({'current_var': current_constant_var_lst})
                var_df['voltage_var'] = voltage_constant_var_lst
            
            # 각 SAMPLE 의 가속 구간의 잔차 분산 추출
            elif label_condition == 'upper':
                
                current_constant_var_lst.append(np.var(main_waveform_data[i][main_waveform_data[i].label == label_condition]['VVVF_SD_IRMS'] - main_waveform_data[i][main_waveform_data[i].label == label_condition]['MOTOR_CURRENT']))
                voltage_constant_var_lst.append(np.var(main_waveform_data[i][main_waveform_data[i].label == label_condition]['VVVF_SD_ES'] - main_waveform_data[i][main_waveform_data[i].label == label_condition]['VVVF_SD_FC']))
                            
                var_df = pd.DataFrame({'current_var': current_constant_var_lst})
                var_df['voltage_var'] = voltage_constant_var_lst
                
            
        var_df = pd.DataFrame({'current_var': current_constant_var_lst})
        var_df['voltage_var'] = voltage_constant_var_lst
        
        return var_df
    
    '''
    전류 전압 분산 데이터 프레임
    '''

    def extract_constant_var_dataframe(self, dict_data, label_condition):
        
        current_constant_var_lst = []
        voltage_constant_var_lst = []
        
        for i in range(len(dict_data)):
            current_constant_var_lst.append(np.var(dict_data[i][dict_data[i].label == label_condition]['VVVF_SD_IRMS'] - dict_data[i][dict_data[i].label == label_condition]['MOTOR_CURRENT']))
            voltage_constant_var_lst.append(np.var(dict_data[i][dict_data[i].label == label_condition]['VVVF_SD_ES'] - dict_data[i][dict_data[i].label == label_condition]['VVVF_SD_FC']))
            
        var_df = pd.DataFrame({'current_var': current_constant_var_lst})
        var_df['voltage_var'] = voltage_constant_var_lst
        
        return var_df
    
    '''
    전류 전압 표준편차 데이터 프레임
    '''

    def extract_constant_std_dataframe(self, dict_data, label_condition):
        
        current_constant_std_lst = []
        voltage_constant_std_lst = []
        
        for i in range(len(dict_data)):
            current_constant_std_lst.append(np.std(dict_data[i][dict_data[i].label == label_condition]['VVVF_SD_IRMS'] - dict_data[i][dict_data[i].label == label_condition]['MOTOR_CURRENT']))
            voltage_constant_std_lst.append(np.std(dict_data[i][dict_data[i].label == label_condition]['VVVF_SD_ES'] - dict_data[i][dict_data[i].label == label_condition]['VVVF_SD_FC']))
            
        std_df = pd.DataFrame({'current_std': current_constant_std_lst})
        std_df['voltage_std'] = voltage_constant_std_lst
        
        return std_df
      
    
if __name__ == "__main__":
    
    r = Extract_Rotem_srs_Data()
    
    '''
    EDA 할 때, 사용한 데이터
    '''
    
    df = r.load_data(path)
    df = r.preprocessed_feature_type(df) # raw data
    df2 = r.preprocessed_df(df) # 이상치 제거한 df
    dict_df = r.get_label(df2) # 가동모드 label 부여 df 
    main_df = r.extract_main_waveform_dataframe(dict_df) # 대표 파형 df
    
    constant_df = r.extract_constant_speed_dataframe(main_df) # 정속 구간 df
    upper_df = r.extract_upper_speed_dataframe(main_df) # 대표 파형 중에 가속만 추출
    
    # 잔차의 분산 df
    res_df_constant = r.extract_speed_range_var_dataframe(main_df, 'constant')
    res_df_upper = r.extract_speed_range_var_dataframe(main_df, 'upper')
    
    # 잔차 df
    res_df = r.extract_residual_dataframe(constant_df,    # editable : constant_df / upper_df
                                          5) 
    
    '''
    24년도 데이터 (검증 데이터)
    '''
    
    df = r.load_data(path2)
    df = r.preprocessed_feature_type(df) # raw data
    df2 = r.preprocessed_df(df) # 이상치 제거한 df
    dict_df = r.get_label(df2) # 가동모드 label 부여 df 
    main_df = r.extract_main_waveform_dataframe(dict_df) # 대표 파형 df

    constant_df = r.extract_constant_speed_dataframe(main_df) # 정속 구간 df
    upper_df = r.extract_upper_speed_dataframe(main_df) # 대표 파형 중에 가속만 추출

    # 잔차의 분산 df
    res_df_constant = r.extract_speed_range_var_dataframe(main_df, 'constant')
    res_df_upper = r.extract_speed_range_var_dataframe(main_df, 'upper')

    # 잔차 df
    res_df = r.extract_residual_dataframe(upper_df,    # editable : constant_df / upper_df
                                          5)
        
'''
딕셔너리 df concat 필요 시
'''

def concat_dict_df(dict_df):
    concat_df = pd.DataFrame()

    for i in range(len(dict_df)):
        concat_df = pd.concat([concat_df, dict_df[i]])
        
    return concat_df
    
    
        