#  Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def get_max_class(row):
    return row.idxmax()

def OCT_eval(df, n='normal', d='drusen', c='cnv', cm_plot=False, drop=False):
    
    df['pred_label'] = df.loc[:,[n, d, c]].apply(get_max_class, axis=1)
    df['pred_label'] = df.pred_label.map({n:0, d:1, c:2})
    
#     print(classification_report(df.label, df.pred_label))
#     print('========================================================')
    
#     df["id"] = df["p_id"].astype(str) + '_' + df["eye"]
    
    # Add three columns to track the number of predictions for each class in a volume
    df_counts = df.groupby(['id', 'pred_label']).size().reset_index(name='counts')
    df_pivot = df_counts.pivot(index='id', columns='pred_label', values='counts').fillna(0)
    df_pivot.columns = ['pred_normal', 'pred_drusen', 'pred_cnv']

    # Add three columns to indicate the count of each class in a volume
    df_counts2 = df.groupby(['id', 'label']).size().reset_index(name='counts')
    df_pivot2 = df_counts2.pivot(index='id', columns='label', values='counts').fillna(0)
    df_pivot2.columns = ['label_normal', 'label_drusen', 'label_cnv']

    # Merge all dataframes and remove unnecessary columns
    df_agg = df.groupby('id').max()

    df_mg = pd.merge(df_pivot, df_pivot2, left_index=True, right_index=True)
    df_mg = pd.merge(df_mg, df_agg, left_index=True, right_index=True)

    df_mg.rename(columns={'pred_label': 'pred_class'}, inplace=True)
#     df_mg.drop(['p_id', 'eye', 'bscan', 'label', 'directory', 'normal', 'drusen', 'cnv'], axis=1, inplace=True)
#     df_mg = df_mg.astype(int)
    if drop:
        df_mg.loc[((df_mg.pred_drusen == 1) & (df_mg.pred_class == 1)), 'pred_class'] = 0
        df_mg.loc[((df_mg.pred_cnv == 1) & (df_mg.pred_class == 2)), 'pred_class'] = 0
    
    print(classification_report(df_mg['class'], df_mg['pred_class']))
    
    if cm_plot:
        conf_matrix = confusion_matrix(df_mg['class'], df_mg['pred_class'])
        class_labels_replacement = ['Normal', 'Drusen', 'CNV']



        # Plot confusion matrix as a heatmap
        plt.figure(figsize=(6, 6), dpi=300)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels_replacement, yticklabels=class_labels_replacement)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
#         plt.title("Confusion Matrix")
        plt.show()
        
    
    return df_mg

df_e = pd.read_csv('final_df_efficientnetb3.csv').drop('Unnamed: 0', axis=1)
df_a = pd.read_csv('final_df_efficientnetb3_attention.csv').drop('Unnamed: 0', axis=1)
df_r = pd.read_csv('final_df_resnet18.csv').drop('Unnamed: 0', axis=1)

# df_r['id'] = df_r['Patient ID'].astype(str) + '_' + df_r['Eye']
df_r['id'] = df_r['Directory'].apply(lambda x: re.sub(r'/[^/]*$', '', x).replace('/', '_'))

df_r.drop(['Patient ID', 'Directory', 'Eye'], axis=1, inplace=True)
df_r.columns = ['class', 'bscan', 'label', 'normal_r', 'drusen_r', 'cnv_r', 'id']

df_e = pd.read_csv('final_df_efficientnetb3.csv').drop('Unnamed: 0', axis=1)
df_e.drop(['Patient ID', 'Directory', 'Eye', 'B-scan', 'Class', 'Label'], axis=1, inplace=True)
df_e.columns = ['normal_e', 'drusen_e', 'cnv_e']

df_a = pd.read_csv('final_df_efficientnetb3_attention.csv').drop('Unnamed: 0', axis=1)
df_a.drop(['Patient ID', 'Directory', 'Eye', 'B-scan', 'Class', 'Label'], axis=1, inplace=True)
df_a.columns = ['normal_a', 'drusen_a', 'cnv_a']

df = pd.concat([df_r, df_e, df_a], axis=1).round(2)

column_order = ['id', 'bscan', 'class', 'label', 'normal_r', 'normal_e', 'normal_a', 'drusen_r', 'drusen_e', 'drusen_a', 'cnv_r', 'cnv_e', 'cnv_a']
df = df[column_order]

dfr = OCT_eval(df, n='normal_r', d='drusen_r', c='cnv_r', cm_plot=True)
dfe = OCT_eval(df, n='normal_e', d='drusen_e', c='cnv_e', cm_plot=True)
dfa = OCT_eval(df, n='normal_a', d='drusen_a', c='cnv_a', cm_plot=True)

df['normal_avg'] = (df.normal_r + df.normal_e + df.normal_a) / 3
df['drusen_avg'] = (df.drusen_r + df.drusen_e + df.drusen_a) / 3
df['cnv_avg'] = (0.4 * df.cnv_r + 0.5 * df.cnv_e + 0.6 * df.cnv_a) / 3
df.sample()

dfens = OCT_eval(df, n='normal_avg', d='drusen_avg', c='cnv_avg', cm_plot=True, drop=False)

#define the true labels and the predicted probabilities
y_true = df["label"]
y_probas = df[["normal_avg", "drusen_avg", "cnv_avg"]]

#import scikitplot to plot the roc curve
import scikitplot as skplt

#plot the roc curve for all classes
class_names = ["Normal", "Drusen", "CNV"]
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()