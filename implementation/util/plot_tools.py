import matplotlib.pyplot as plt
import numpy as np
import os

# Image specific tools #
def show_and_save(output: list, title: str, filename: str):
    plt.imshow(output)
    plt.figtext(0.2, 0.01, format(title), wrap=True)
    plt.savefig("{}.png".format(filename))
    plt.show()

# Series specific tools #
def save_series_plot(series, type, file_dir, file_name, clear):
    x_axis = np.linspace(1, len(series)+1, len(series))
    plt.title("Average {} per epoch".format(type))
    plt.xlabel("Epoch")
    plt.ylabel("Average {}".format(type))
    plt.plot(x_axis, series)
    plt.savefig("{}.png".format(os.path.join(file_dir, file_name)))
    if clear:
        plt.clf()

def show_and_save_series_plot(series, type, file_dir, file_name):
    save_series_plot(series, type, file_dir, file_name, False)
    plt.show()