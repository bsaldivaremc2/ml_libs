from ml_libs.util import stats
def draw_confidence_intervals(idf,confidence_level=0.95,point_marker='x',point_color='red',
                             mean_line_color='red',
                              box_alpha=0.2,box_color='blue',box_width=1,
                              column_separation=2,
                              y_resolution=0.5,max_y=None,
                              show_mean=True,
                              draw_th_line=True,
                              th_line_color='red',
                              th_line_y = 0,
                              th_line_alpha = 0.5,
                              grid_kargs={'b':None,'which':'major','axis':'both','color':'grey','alpha':0.2},
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
    upper_y = maxs+y_resolution
    if type(max_y) in [float,int]:
    	if max_y<=1:
    		upper_y = max_y
    yvals = np.arange(mins-y_resolution,upper_y,y_resolution).round(2)
    plt.yticks(yvals,yvals)
    #
    plt.ylabel(ylabel)
    plt.grid(**grid_kargs)
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


def draw_confidence_intervals_2_distributions(idf1,idf2,confidence_level=0.95,
                                              point_markers=['x','o'],point_colors=['red','blue'],
                                              labels=['not random','random'],
                             mean_line_color='red',
                              box_alpha=0.2,box_color='blue',box_width=1,box_separation=0.05,
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
    import pandas as pd
    def mix_arrays(inp1,inp2):
        o = []
        for _ in range(inp1.flatten().shape[0]):
            o.append(inp1[_])
            o.append(inp2[_])
        return np.array(o).copy()
    def get_x_y_scatter(inp,xpos,x_offset=0,y_offset=0):
        xs = []
        ys = []
        for _ in range(inp.shape[1]):
            x = xpos[_]
            for v in range(inp.shape[0]):
                y = inp[v,_]
                xs.append(x+x_offset)
                ys.append(y+y_offset)
        return xs.copy(),ys.copy()
    values1, values2 = idf1.values,idf2.values
    values = np.hstack([values1,values2])
    cols = list(idf1.columns)
    colsn = len(cols)
    #Means
    means1 = idf1.mean(0).values
    means2 = idf2.mean(0).values
    means = mix_arrays(means1,means2)
    #Stds
    stds1 = idf1.std(0).values
    stds2 = idf2.std(0).values
    stds = mix_arrays(stds1,stds2)
    maxs = values.max()
    mins = values.min()
    n = idf1.shape[0]
    df = n - 1
    confidence_intervals = stats.get_confidence_interval(stds,n,df,confidence_level)
    lower = means - confidence_intervals
    upper = means + confidence_intervals
    xvals = np.arange(0,colsn*column_separation,column_separation)
    left_offset = -(box_width*(1+box_separation))
    right_offset = box_width*box_separation
    #Draw
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
        y1 = lower[_*2]
        h1 = upper[_*2] - y1
        y2 = lower[_*2+1]
        h2 = upper[_*2+1] - y2
        #X
        x1 = xvals[_]+left_offset
        x2 = xvals[_]+right_offset
        rect = patches.Rectangle((x1,y1),box_width,h1,fill=True,linewidth=1,color=box_color,alpha=box_alpha)
        plt.gca().add_patch(rect)
        rect = patches.Rectangle((x2,y2),box_width,h2,fill=True,linewidth=1,color=box_color,alpha=box_alpha)
        plt.gca().add_patch(rect)
        if show_mean==True:
            for x,y,h in zip([x1,x2],[y1,y2],[h1,h2]):
                xs = x
                xe = x+box_width
                y = y + h/2
                plt.plot([xs,xe],[y,y],color=mean_line_color)
    all_values = [values1,values2]
    offsets = [left_offset+box_width/2,right_offset+box_width/2]
    for v,offset,point_color,point_marker,label in zip(all_values,offsets,point_colors,point_markers,labels):
        xs, ys = get_x_y_scatter(v,xvals,x_offset=offset,y_offset=0)
        plt.scatter(xs,ys,color=point_color,marker=point_marker,label=label)
    plt.xticks(xvals,cols,rotation=90)
    plt.xlabel(xlabel)
    #
    lower_y = np.floor(mins-y_resolution)
    upper_y = np.ceil(maxs+y_resolution)
    yvals = np.arange(lower_y,upper_y,y_resolution).round(2)
    plt.yticks(yvals,yvals)
    #
    plt.ylabel(ylabel)
    plt.legend()
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

