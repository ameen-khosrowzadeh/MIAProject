import argparse
import os

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ptitprince as pt




def get_newest_folder(path):
    newest = None
    date = None

    for f in get_all_folders(path):
        if (date == None or date < os.path.getmtime(path+f)):
            newest = f
            date = os.path.getmtime(path+f)

    return os.path.join(path, newest)


def get_all_folders(path):
    return [x for x in os.listdir(path) if os.path.isfile(x) == False]


def set_box_format(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['caps'], linewidth=0)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=1.5)
    plt.setp(bp['fliers'], marker='.')
    plt.setp(bp['fliers'], markerfacecolor='black')
    plt.setp(bp['fliers'], alpha=1)


def boxplot(file_path: str, data: list, title: str, x_label: str, y_label: str, x_ticks: tuple,
            min_: float = None, max_: float = None):
    if len(data) != len(x_ticks):
        raise ValueError('arguments data and x_ticks need to have same length')

    fig = plt.figure(
        figsize=(4.8*1.5, 6.4 *1.5))  # figsize defaults to (width, height) =(6.4, 4.8),
    # for boxplots, we want the ratio to be inversed
    ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
    bp = ax.boxplot(data, widths=0.6)
    set_box_format(bp, '000')

    # set and format litle, labels, and ticks
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=12)
    # ax.set_xlabel(x_label, fontweight='bold', fontsize=9.5)  # we don't use the x-label since it should be clear from the x-ticks
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticklabels(x_ticks, fontdict={'fontsize': 12, 'fontweight': 'bold'})

    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # thicken frame
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # adjust min and max if provided
    if min_ is not None or max_ is not None:
        min_original, max_original = ax.get_ylim()
        min_ = min_ if min_ is not None and min_ < min_original else min_original
        max_ = max_ if max_ is not None and max_ > max_original else max_original
        ax.set_ylim(min_, max_)

    plt.savefig(file_path)
    plt.close()





def boxplot_per_metric(path,metric):
    results = pd.read_csv(path+'/results.csv', delimiter=';')
    results.boxplot(column=metric, by='LABEL')
    plt.savefig(os.path.join(path,'results',metric+'.png'))
    # plt.show()



def my_rain_plot(path):
    results = pd.read_csv(path + '/results.csv', delimiter=';')
    del results['HDRFDST']
    new_results=results.melt(id_vars=["SUBJECT","LABEL"],var_name="metric",value_name="Score")
    dx = "LABEL";
    dy = "Score";
    dhue = "metric";
    ort = "v";
    pal = "Set2";
    sigma = .2
    f, ax = plt.subplots(figsize=(12, 5))

    ax = pt.RainCloud(x=dx, y=dy, hue=dhue, data=new_results, palette=pal, bw=sigma, width_viol=1,
                      ax=ax, orient=ort, alpha=.65, dodge=True, pointplot=True, move=.1)
    plt.show()


def format_data(data, label: str, metric: str):
    return data[data['LABEL'] == label][metric].values


def metric_to_readable_text(metric: str):
    if metric == 'DICE':
        return 'Dice coefficient'
    elif metric == 'HDRFDST':
        return 'Hausdorff distance (mm)'
    else:
        # raise Warning('Metric "{}" unknown'.format(metric))
        return 'unknown'


def main(csv_file: str, plot_dir: str):
    metrics = ('DICE', 'HDRFDST')  # the metrics we want to plot the results for
    # metrics = ('DICE', 'HDRFDST', 'ACURCY','KAPPA')
    # metrics = ('DICE', 'HDRFDST', 'ACURCY', 'VOLSMTY', 'PRCISON', 'MUTINF', 'KAPPA', 'JACRD', 'GCOERR')


    # metrics= ('ACURCY', 'AUC' , 'DICE' , 'FN' , 'FP' , 'HDRFDST' , 'JACRD' , 'KAPPA' ,'MUTINF' , 'PRCISON', 'SNSVTY' , 'SPCFTY' , 'TN' , 'TP' , 'VOLSMTY' )
    metrics_yaxis_limits = ((0.0, 1.0), (0.0, None))  # tuples of y-axis limits (min, max) for each metric. Use None if unknown
    labels = ('WhiteMatter', 'Amygdala')  # the brain structures/tissues you are interested in


    last = get_newest_folder('./mia-result/')
    plot_dir = os.path.join(last, 'results')
    os.makedirs(plot_dir, exist_ok=True)

    csv_file = last + '/results.csv'
    print(csv_file)

    # for metric in metrics:
    #     boxplot_per_metric(last,metric)
    my_rain_plot(last)

    # load the CSVs. We usually want to compare different methods (e.g. a set of different features), therefore,
    # we load two CSV (for simplicity, it is the same here)
    # # todo: adapt to your needs to compare different methods (e.g. load different CSVs)
    # df_method1 = pd.read_csv(csv_file, sep=';')
    # df_method2 = pd.read_csv(csv_file, sep=';')
    # dfs = [df_method1, df_method2]
    #
    # # some parameters to improve the plot's readability
    # methods = ('Method 1', 'Method 2')
    # title = 'Your experiment comparing method 1 and 2 on {}'
    #
    #
    # for label in labels:
    #     for metric, (min_, max_) in zip(metrics, metrics_yaxis_limits):
    #         boxplot(os.path.join(plot_dir, '{}_{}.png'.format(label, metric)),
    #                 [format_data(df, label, metric) for df in dfs],
    #                 title.format(label),
    #                 'Method', metric_to_readable_text(metric),
    #                 methods,
    #                 min_, max_
    #                 )


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')

    parser.add_argument(
        '--csv_file',
        type=str,
        default='../results/results.csv',
        help='Path to the result CSV file.'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='../results',
        help='Path to the plot directory.'
    )

    args = parser.parse_args()
    main(args.csv_file, args.plot_dir)
