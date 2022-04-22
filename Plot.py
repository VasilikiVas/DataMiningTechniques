import pandas as pd
import matplotlib.pyplot as plt


def plot_bar(error_dict, name, color=None):
    df = pd.DataFrame(error_dict)
    print(df)
    if color is not None:
        ax = df.plot.bar(x='Method', rot=0, color = color)
    else:
        ax = df.plot.bar(x='Method', rot=0)

    # for container in ax.containers:
    #      ax.bar_label(container)

    ax.set_ylabel("Mean absolute error")
    ax.set_ylim([0.4, 0.52])
    # ax.set_xlabel("Experiment")

    plt.tight_layout()
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    error_dict = {'Method': ['SVR','LSTM'],
                  'Window size 2': [0.468,0.487],
                  'Window size 3': [0.455,0.489],
                  'Window size 4': [0.459,0.476]}
    plot_bar(error_dict, 'mae_barplot_windowsize.png')
    error_dict = {'Method': ['SVR','LSTM'],
                  'Combination': [0.444,0.468],
                  'No combination': [0.455,0.489]}
    plot_bar(error_dict, 'mae_barplot_combining_features.png')
    error_dict = {'Method': ['SVR','LSTM'],
                  'No additional features': [0.455, 0.489],
                  'Sleep (added)': [0.454, 0.495],
                  'Weekend (added)': [0.448, 0.474],
                  'appCat.builtin (removed)': [0.4524,0.490],
                  'appCat.social (removed)': [0.463,0.489],
                  'Call (removed)' : [0.454,0.494],
                  'Sms (removed)' : [0.463,0.490]
                  }
    plot_bar(error_dict, 'mae_barplot_feature_analysis.png',['b', '#19a426', '#68e875', '#9b1717', '#ba1c1c', '#df2a2a', '#e96d6d'])
