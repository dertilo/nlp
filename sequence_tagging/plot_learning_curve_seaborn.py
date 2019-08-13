import pandas as pd
from commons import data_io

if __name__ == '__main__':
    import seaborn as sns

    sns.set(style="ticks", palette="pastel")
    file = '/home/tilo/hpc/data/scierc_data/learning_curve_scores.json'
    d = data_io.read_jsons_from_file_to_list(file)[0]
    data = [{'train_size':round(float(train_size),2),'f1-spanwise':score[traintest]['f1-spanwise'],'traintest':traintest} for train_size,scores in d.items() for score in scores for traintest in ['train','test']]
    df = pd.DataFrame(data=data)

    ax = sns.boxplot(x="train_size", y="f1-spanwise",
                     hue="traintest", palette=["m", "g"],
                     data=df)
    # sns.despine(offset=10, trim=True)
    ax.set_title('spacy+crfsuite with 3-fold-crossval')
    ax.figure.savefig("/tmp/seaborn_plot.png")