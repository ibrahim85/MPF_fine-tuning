
import numpy as np
import matplotlib.pyplot as plt
import PIL
import matplotlib.colors as colors


def displayNetwork(A,optNormEach=False,optNormAll=True,numColumns=None,imageWidth=None,cmapName='gray',
                   borderColor='black',borderWidth=1,verbose=True,graphicsLibrary='matplotlib',saveName=''):
    # This function visualizes filters in matrix A. Each row of A is a
    # filter. We will reshape each row into a square image and visualizes
    # on each cell of the visualization panel. All other parameters are
    # optional, usually you do not need to worry about them.
    #
    # optNormEach: whether we need to normalize each row so that
    # the mean of each row is zero.
    #
    # optNormAll: whether we need to normalize all the rows so that
    # the mean of all the rows together is zero.
    #
    # imageWidth: how many pixels are there for each image
    # Default value is the squareroot of the number of columns in A.
    #
    # numColumns: how many columns are there in the display.
    # Default value is the squareroot of the number of rows in A.

    # compute number of rows and columns
    nr,nc = np.shape(A)
    if imageWidth==None:
        sx = np.ceil(np.sqrt(nc))
        sy = np.ceil(nc/sx)
    else:
        sx = imageWidth
        sy = np.ceil(nc/sx)
    if numColumns==None:
        n = np.ceil(np.sqrt(nr))
        m = np.ceil(nr/n)
    else:
        n = numColumns
        m = np.ceil(nr/n)
    n = np.uint8(n)
    m = np.uint8(m)
    if optNormAll:
        A = A-A.min()
        A = A/A.max()

    # insert data onto squares on the screen
    k = 0
    buf = borderWidth
    array = -np.ones([buf+m*(sy+buf),buf+n*(sx+buf)])
    for i in range(1,m+1):
        for j in range(1,n+1):
            if k>=nr:
                continue
            B = A[k,:]
            if optNormEach:
                B = B-B.min()
                B = B/float(B.max())
            B = np.reshape(np.concatenate((B,-np.ones(sx*sy-nc)),axis=0),(sx,-1))
            array[(buf+(i-1)*(sy+buf)):(i*(sy+buf)),(buf+(j-1)*(sx+buf)):(j*(sx+buf))] = B
            k = k+1

    # display picture and save it
    cmap = plt.cm.get_cmap(cmapName)
    cmap.set_under(color=borderColor)
    cmap.set_bad(color=borderColor)
    if graphicsLibrary=='PIL':
        im = PIL.Image.fromarray(np.uint8(cmap(array)*255))
        if verbose:
            im.show()
        if saveName != '':
            im.save(saveName)
    elif graphicsLibrary=='matplotlib':
        plt.imshow(array,interpolation='nearest',
                   norm=colors.Normalize(vmin=0.0,vmax=1.0,clip=False))
        plt.set_cmap(cmap)
        # if verbose:
        #     #plt.show()
        if saveName != '':
            plt.savefig(saveName)