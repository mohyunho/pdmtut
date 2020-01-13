import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

### Data assignment

# read csv file and assign data into pandas DataFrame
telemetry = pd.read_csv('PdM_ML/data/PdM_telemetry.csv')
errors = pd.read_csv('PdM_ML/data/PdM_errors.csv')
maint = pd.read_csv('PdM_ML/data/PdM_maint.csv')
failures = pd.read_csv('PdM_ML/data/PdM_failures.csv')
machines = pd.read_csv('PdM_ML/data/PdM_machines.csv')


'''
Telemetry:
The first data source is the telemetry time-series data which consists of 
voltage, rotation, pressure, and vibration measurements 
collected from 100 machines in real time 
averaged over every hour collected during the year 2015
col : datetime | machineID | volt | rotate | pressure | vibration

'''
# format datetime field which comes in as string
print ("telemetry['datetime']", telemetry['datetime'])


telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
print ("telemetry['datetime']", telemetry['datetime'])


print("Total number of telemetry records: %d" % len(telemetry.index))
print(telemetry.head())

print ("size:", telemetry.size)
print ("ndim:", telemetry.ndim)

sns.set_style("darkgrid")
plt.figure(figsize=(25, 12))

tel_machine = []
for i in range(20):
    print (i)
    plot_df = telemetry.loc[(telemetry['machineID'] == i+1) &
                        (telemetry['datetime'] > pd.to_datetime('2015-01-01')) &
                        (telemetry['datetime'] <pd.to_datetime('2015-02-01')),
                        ['datetime','volt']]
    tel_machine.append(plot_df)
    plt.subplot(5,4,i+1)
    plt.plot(plot_df['datetime'], plot_df['volt'])

    if i == 0:
        plt.ylabel('voltage')
        plt.xticks([])
    elif i != 19 :
        plt.xticks([])
        plt.yticks([])
    else :
        adf = plt.gca().get_xaxis().get_major_formatter()
        adf.scaled[1.0] = '%m-%d-%Y'
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.yticks([])
    plt.title('Machine %s' %(i+1))


'''
Errors:
The second major data source is the error logs. 
These are non-breaking errors thrown while the machine is still operational 
and do not constitute as failures. 
The error date and times are rounded to the closest hour 
since the telemetry data is collected at an hourly rate
col : datetime | machineID | errorID

'''
# format of datetime field which comes in as string
errors['datetime'] = pd.to_datetime(errors['datetime'],format = '%Y-%m-%d %H:%M:%S')
errors['errorID'] = errors['errorID'].astype('category')
print("Total Number of error records: %d" %len(errors.index))
print(errors.head())

sns.set_style("darkgrid")
plt.figure(figsize=(20, 8))
ax = errors['machineID'].value_counts().plot(kind='bar')
ax.set_xlim(-0.5, 10.5)
plt.ylabel('Count')
errors['machineID'].value_counts()


'''
Maintenance:
These are the scheduled and unscheduled maintenance records which correspond to both regular inspection of components 
as well as failures. A record is generated if a component is replaced during the scheduled inspection or 
replaced due to a breakdown. The records that are created due to breakdowns will be called failures 
which is explained in the later sections. Maintenance data has both 2014 and 2015 records.
col : datetime | machineID | comp
'''

maint['datetime'] = pd.to_datetime(maint['datetime'], format='%Y-%m-%d %H:%M:%S')
maint['comp'] = maint['comp'].astype('category')
print("Total Number of maintenance Records: %d" %len(maint.index))
print(maint.head())



'''
Machines:
This data set includes some information about the machines: model type and age (years in service).
col : machineID | model | age
'''
machines['model'] = machines['model'].astype('category')
print("Total number of machines: %d" % len(machines.index))
print(machines.head())





'''
Failures:
These are the records of component replacements due to failures. 
Each record has a date and time, machine ID, and failed component type.
col : datetime | machineID | failure
'''
failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
# failures['failure'] = failures['failure'].astype('category')
print
print("Total number of failures: %d" % len(failures.index))
print(failures.head())




### Feature extraction (from data sources)

## Calculate mean/std values for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='mean').unstack())
telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
telemetry_mean_3h.reset_index(inplace=True)

# repeat for standard deviation
temp = []
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='std').unstack())
telemetry_sd_3h = pd.concat(temp, axis=1)
telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
telemetry_sd_3h.reset_index(inplace=True)

print (telemetry_mean_3h.head())

temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,      index='datetime',
                                               columns='machineID',
                                               values=col).rolling(window=24,center=False).mean().resample('3H',
                                                                                closed='left',
                                                                                label='right',
                                                                                how='first').unstack())
telemetry_mean_24h = pd.concat(temp, axis=1)
telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
telemetry_mean_24h.reset_index(inplace=True)
telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

# repeat for standard deviation
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry, index='datetime',
                                          columns='machineID',
                                          values=col).rolling(window=24,center=False).mean().resample('3H',
                                                                                closed='left',
                                                                                label='right',
                                                                                how='first').unstack())
telemetry_sd_24h = pd.concat(temp, axis=1)
telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
telemetry_sd_24h.reset_index(inplace=True)

# Notice that a 24h rolling average is not available at the earliest timepoints
print (telemetry_mean_24h.head(20))

# merge columns of feature sets created earlier
telemetry_feat = pd.concat([telemetry_mean_3h,
                            telemetry_sd_3h.ix[:, 2:6],
                            telemetry_mean_24h.ix[:, 2:6],
                            telemetry_sd_24h.ix[:, 2:6]], axis=1).dropna()

## Feature extraction from errors data (error IDs are categorical values
# and should not be averaged over time intervals like the telemetry measurements)

# create a column for each error type
error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
error_count
error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
print(error_count.head(13))
# combine errors for a given machine in a given hour
error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
print(error_count.head(13))

error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)

temp = []
fields = ['error%d' % i for i in range(1,6)]
for col in fields:
    temp.append(pd.pivot_table(error_count,    index='datetime',
                                               columns='machineID',
                                               values=col).rolling(window=24,center=False).sum().resample('3H',
                                                                             closed='left',
                                                                             label='right',
                                                                             how='first').unstack())
error_count = pd.concat(temp, axis=1)
error_count.columns = [i + 'count' for i in fields]
error_count.reset_index(inplace=True)
error_count = error_count.dropna()
print(error_count.head(13))




#days since most recent component change
# create a column for each error type
comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

# combine repairs for a given machine in a given hour
comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

# add timepoints where no components were replaced
comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                      on=['datetime', 'machineID'],
                                                      how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

components = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in components:
    # convert indicator to most recent date of component change
    comp_rep.loc[comp_rep[comp] < 1, comp] = None
    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']

    # forward-fill the most-recent date of component change
    comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

print ("comp_rep['comp1']", comp_rep['comp1'])

# remove dates in 2014 (may have NaN or future component change dates)
comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]


# replace dates of most recent component change with days since most recent component change
for comp in components:
    comp_rep[comp] = pd.Series(comp_rep[comp])
    comp_rep[comp] = pd.to_datetime(comp_rep[comp])
    comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')

print(comp_rep.head(20))


## Merge all features
final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(machines, on=['machineID'], how='left')

print(final_feat.head())



### Label construction
labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
print (labeled_features.head(1000))
print (labeled_features['failure'])

labeled_features['failure'] = labeled_features['failure'].fillna(method='bfill', limit=7) # fill backward up to 24h
labeled_features['failure'] = labeled_features['failure'].fillna('none')
labeled_features.head()
print (labeled_features.head(1000))



### ML training and test

# make test and training splits
threshold_dates = [[pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')],
                   [pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')],
                   [pd.to_datetime('2015-09-30 01:00:00'), pd.to_datetime('2015-10-01 01:00:00')]]

test_results = []
models = []
for last_train_date, first_test_date in threshold_dates:
    print ("last_train_date", last_train_date)
    print ("first_test_date", first_test_date)
    # split out training and test data
    train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
    print ("train_y", train_y)
    train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
                                                                                                        'machineID',
                                                                                                        'failure'], 1))

    print ("train_X", train_X)
    test_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date].drop(['datetime',
                                                                                                       'machineID',
                                                                                                       'failure'], 1))


# train and predict using the model, storing results for later
filename = 'GBC_pdm_model.sav'
try:
    f = open(filename)
    # Do something with the file
    print("Trained model already exists")
    my_model = pickle.load(open(filename, 'rb'))

except IOError:
    # print("File not accessible")
    my_model = GradientBoostingClassifier(random_state=42, learning_rate=0.01, n_estimators=100, verbose=1)
    print("Training in progress...")
    my_model.fit(train_X, train_y)
    # save trained model
    pickle.dump(my_model, open(filename, 'wb'))


test_result = pd.DataFrame(labeled_features.loc[labeled_features['datetime'] > first_test_date])
test_result['predicted_failure'] = my_model.predict(test_X)
print ("test_result", test_result)
test_results.append(test_result)
models.append(my_model)


export_test_csv = test_results[0].to_csv('test_result_GBC.csv')

sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
labels, importances = zip(*sorted(zip(test_X.columns, models[0].feature_importances_), reverse=True, key=lambda x: x[1]))
plt.xticks(range(len(labels)), labels)
_, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.bar(range(len(importances)), importances)
plt.ylabel('Importance')



plt.show()



from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score


def Evaluate(predicted, actual, labels):
    output_labels = []
    output = []

    # Calculate and display confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
    print(cm)

    # Calculate precision, recall, and F1 score
    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
    precision = precision_score(actual, predicted, average=None, labels=labels)
    recall = recall_score(actual, predicted, average=None, labels=labels)
    f1 = 2 * precision * recall / (precision + recall)
    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
    output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])

    # Calculate the macro versions of these metrics
    output.extend([[np.mean(precision)] * len(labels),
                   [np.mean(recall)] * len(labels),
                   [np.mean(f1)] * len(labels)])
    output_labels.extend(['macro precision', 'macro recall', 'macro F1'])

    # Find the one-vs.-all confusion matrix
    cm_row_sums = cm.sum(axis=1)
    cm_col_sums = cm.sum(axis=0)
    s = np.zeros((2, 2))
    for i in range(len(labels)):
        v = np.array([[cm[i, i],
                       cm_row_sums[i] - cm[i, i]],
                      [cm_col_sums[i] - cm[i, i],
                       np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])
        s += v
    s_row_sums = s.sum(axis=1)

    # Add average accuracy and micro-averaged  precision/recall/F1
    avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)
    micro_prf = [float(s[0, 0]) / s_row_sums[0]] * len(labels)
    output.extend([avg_accuracy, micro_prf])
    output_labels.extend(['average accuracy',
                          'micro-averaged precision/recall/F1'])

    # Compute metrics for the majority classifier
    mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]
    cm_row_dist = cm_row_sums / float(np.sum(cm))
    mc_accuracy = 0 * cm_row_dist
    mc_accuracy[mc_index] = cm_row_dist[mc_index]
    mc_recall = 0 * cm_row_dist
    mc_recall[mc_index] = 1
    mc_precision = 0 * cm_row_dist
    mc_precision[mc_index] = cm_row_dist[mc_index]
    mc_F1 = 0 * cm_row_dist
    mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)
    output.extend([mc_accuracy.tolist(), mc_recall.tolist(),
                   mc_precision.tolist(), mc_F1.tolist()])
    output_labels.extend(['majority class accuracy', 'majority class recall',
                          'majority class precision', 'majority class F1'])

    # Random accuracy and kappa
    cm_col_dist = cm_col_sums / float(np.sum(cm))
    exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))
    kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)
    output.extend([exp_accuracy.tolist(), kappa.tolist()])
    output_labels.extend(['expected accuracy', 'kappa'])

    # Random guess
    rg_accuracy = np.ones(len(labels)) / float(len(labels))
    rg_precision = cm_row_dist
    rg_recall = np.ones(len(labels)) / float(len(labels))
    rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)
    output.extend([rg_accuracy.tolist(), rg_precision.tolist(),
                   rg_recall.tolist(), rg_F1.tolist()])
    output_labels.extend(['random guess accuracy', 'random guess precision',
                          'random guess recall', 'random guess F1'])

    # Random weighted guess
    rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist ** 2)
    rwg_precision = cm_row_dist
    rwg_recall = cm_row_dist
    rwg_F1 = cm_row_dist
    output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),
                   rwg_recall.tolist(), rwg_F1.tolist()])
    output_labels.extend(['random weighted guess accuracy',
                          'random weighted guess precision',
                          'random weighted guess recall',
                          'random weighted guess F1'])

    output_df = pd.DataFrame(output, columns=labels)
    output_df.index = output_labels

    return output_df


evaluation_results = []
for i, test_result in enumerate(test_results):
    print('\nSplit %d:' % (i + 1))
    evaluation_result = Evaluate(actual=test_result['failure'],
                                 predicted=test_result['predicted_failure'],
                                 labels=['none', 'comp1', 'comp2', 'comp3', 'comp4'])
    evaluation_results.append(evaluation_result)
print (evaluation_results[0])  # show full results for first split only



print ("end of line")