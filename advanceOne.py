from pandas import read_csv
filename = '../dataset/csv/2015_data.csv'
names = ['resident_status', 'education_1989_revision', 'education_2003_revision',
         'education_reporting_flag', 'month_of_death', 'sex,detail_age_type', 'detail_age',
         'age_substitution_flag', 'age_recode_52','age_recode_27','age_recode_12','infant_age_recode_22',
         'place_of_death_and_decedents_status','marital_status','day_of_week_of_death',
         'current_data_year','injury_at_work','manner_of_death','method_of_disposition',
         'autopsy','activity_code','place_of_injury_for_causes_w00_y34_except_y06_and_y07_',
         'icd_code_10th_revision','358_cause_recode','113_cause_recode','130_infant_cause_recode',
         '39_cause_recode','number_of_entity_axis_conditions','entity_condition_1','entity_condition_2',                                                                                                                                                                                                                                                                                                                                                                                                                                
         'entity_condition_3','entity_condition_4','entity_condition_5','entity_condition_6',
         'entity_condition_7','entity_condition_8','entity_condition_9','entity_condition_10',
         'entity_condition_11','entity_condition_12','entity_condition_13','entity_condition_14',
         'entity_condition_15','entity_condition_16','entity_condition_17','entity_condition_18',
         'entity_condition_19','entity_condition_20','number_of_record_axis_conditions',
         'record_condition_1','record_condition_2','record_condition_3','record_condition_4',
        'record_condition_5','record_condition_6','record_condition_7','record_condition_8',
        'record_condition_9','record_condition_10','record_condition_11','record_condition_12',
        'record_condition_13','record_condition_14','record_condition_15','record_condition_16',
        'record_condition_17','record_condition_18','record_condition_19','record_condition_20',
        'race','bridged_race_flag','race_imputation_flag','race_recode_3','race_recode_5',
        'hispanic_origin','hispanic_originrace_recode']
data = read_csv(filename, names=names,sep=";",encoding="ISO-8859-1")
print(data.shape)