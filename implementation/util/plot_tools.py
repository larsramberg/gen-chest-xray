import matplotlib.pyplot as plt

def show_and_save(output: list, epoch: int, label: str):
    plt.imshow(output)
    plt.figtext(0.2, 0.01, "Label {} after {} epochs".format(label, epoch), wrap=True)
    plt.savefig('result_{}.png'.format(epoch))
    plt.show()
