import numpy as np
import matplotlib.pyplot as plt
SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

def scatter_(dict_,x_name,y_name,z_name):
  fig, ax = plt.subplots()
  x,y = dict_[x_name],np.log10(dict_[y_name])
  ax.scatter(x,y)
  for i, txt in enumerate(dict_[z_name]):
      ax.annotate("{:.2f}".format(txt), (x[i], y[i]))
  ax.set_xlabel(x_name)
  ax.set_ylabel(y_name)

def scatter_with_colorbar(dict_,x_name,y_name,z_name):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    x,y,z = dict_[x_name],np.log10(dict_[y_name]),dict_[z_name]
    ticks = np.linspace(np.min(z), np.max(z), 5, endpoint=True)
    C = ax.scatter(x=x,y=y,c=z,cmap="coolwarm")
    cb = fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=z_name,ticks=ticks)
    cb.set_label(label=z_name, size=SIZE_LEGEND)
    cb.ax.tick_params(labelsize=SIZE_TICKS)
    plt.xlabel( x_name, fontsize = SIZE_LABELS )
    plt.ylabel( y_name, fontsize = SIZE_LABELS )
    plt.xticks( fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
