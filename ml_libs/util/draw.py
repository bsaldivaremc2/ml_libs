from ml_libs.util import stats
def draw_confidence_intervals(idf,confidence_level=0.95,point_marker='x',point_color='red',
                             mean_line_color='red',
                              box_alpha=0.2,box_color='blue',box_width=1,
                              column_separation=2,
                              y_resolution=0.5,
                              show_mean=True,
                              draw_th_line=True,
                              th_line_color='red',
                              th_line_y = 0,
                              th_line_alpha = 0.5,
                             figsize=(16,9),dpi=80,
                             title='',xlabel='',ylabel='',
                              show=True,
                             save=None,save_kargs={'transparent':False}):
    """
    idf: A dataframe that has the samples as rows and the columns are the variables to draw
    For save_kargs:
    #https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import os
    values = idf.values
    means = idf.mean(0).values
    maxs = values.max()
    mins = values.min()
    stds = idf.std(0).values
    n = idf.shape[0]
    df = n - 1
    confidence_intervals = stats.get_confidence_interval(stds,n,df,confidence_level)
    lower = means - confidence_intervals
    upper = means + confidence_intervals
    cols = list(idf.columns)
    colsn = len(cols)
    xvals = np.arange(0,colsn*column_separation,column_separation)
    #
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.title(title)
    #Draw zero line
    if draw_th_line==True:
        xs = xvals[0]-box_width
        xe = xvals[-1]+box_width
        y = th_line_y
        plt.plot([xs,xe],[y,y],color=th_line_color,alpha=th_line_alpha)
    #Draw boxes
    for _ in range(colsn):
        y = lower[_]
        height = upper[_] - y
        x = xvals[_]-box_width/2
        rect = patches.Rectangle((x,y),box_width,height,fill=True,linewidth=1,color=box_color,alpha=box_alpha)
        plt.gca().add_patch(rect)
        if show_mean==True:
            xs = x
            xe = x+box_width
            y = y + height/2
            plt.plot([xs,xe],[y,y],color=mean_line_color)
    xs = []
    ys = []
    for _ in range(colsn):
        x = xvals[_]
        for v in range(n):
            y = values[v,_]
            xs.append(x)
            ys.append(y)
    plt.scatter(xs,ys,color=point_color,marker=point_marker)
    plt.xticks(xvals,cols,rotation=90)
    plt.xlabel(xlabel)
    #
    yvals = np.arange(mins-y_resolution,maxs+y_resolution,y_resolution).round(2)
    plt.yticks(yvals,yvals)
    #
    plt.ylabel(ylabel)
    if show==True:
        plt.show()
    if type(save)==str:
        path="./"
        if "/" in save:
            path = save.split("/")[:-1]
            path = "/".join(path)
            os.makedirs(path, exist_ok=True)
        fig.savefig(save+".png",dpi=dpi,bbox_inches = "tight",**save_kargs)
    if show!=True:
        plt.close()



def draw_rect_example():
	import matplotlib.pyplot as plt
	import matplotlib.patches as patches
	import numpy as np
	# Create figure and axes
	#fig,ax = plt.subplots(1)
	# Display the image
	# Create a Rectangle patch
	rect = patches.Rectangle((10,50),10,40,linewidth=1,edgecolor='r',facecolor='none') 
	rect = patches.Rectangle((10,50),10,40,fill=True,linewidth=1,color='red',alpha=0.2) 
	#X (col),y (rows), width, height
	x = np.linspace(0,100,100)
	plt.plot(x,x)
	# Add the patch to the Axes
	plt.gca().add_patch(rect)
	plt.show()

	#option 2
	#fig,ax = plt.subplots(1)
	#ax.add_patch(rect)

