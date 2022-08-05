from math import *
from tkinter import *
from App_interface import lens_plot_area


global nxwin, nywin

nxwin, nywin = lens_plot_area()

def Nint(x): return int(floor(x + 0.5))


def Sign(x,y): return (abs(x) if y > 0 else -abs(x))


def Magn(x):
#----------------------------------------------------------------------------
# Returns the order of magnitude of x as 10^n
#----------------------------------------------------------------------------
   return 0e0 if x == 0e0 else \
          10e0**int(log10(abs(x))) if abs(x) >= 1e0 else \
          0.1e0**(int(abs(log10(abs(x))))+1e0)


def Limits(xmin, xmax, maxint = 10):
#-------------------------------------------------------------------------------
#  Replaces the limits xmin and xmax of a real interval with the limits of
#  the smallest extended inteval which includes a number <= maxint of
#  subintervals of length d * 10**p, with d = 1, 2, 5 and p integer.
#
#  scale - scale factor (10**p)
#  nsigd - relevant number of significant digits
#  nintv - number of subintervals
#
#  Returns: xmin, xmax, scale, nsigd, nintv
#-------------------------------------------------------------------------------
    eps = 1e-5                                  #relative precision
    xfact = [0.5e0, 0.5e0, 0.4e0]             #scale factors
    maxinter = 1000                           #maximum number of iterations


    if (abs(xmax-xmin) < 10e0 * eps * abs(xmax)):
        corrmin = 1e0 - 10e0 * Sign(eps, xmin)
        corrmax = 1e0 + 10e0 * Sign(eps, xmax)
        xmin *= corrmin
        xmax *= corrmax
       #Initial factor scale
    factor = 1e0/ (eps * min(Magn(xmin), Magn(xmax))) if (xmin * xmax) else \
             1e0/(eps * max(Magn(xmin), Magn(xmax)))

#Corrections

    corrmin = 1e0 + Sign(eps, xmin)
    corrmax = 1e0 - Sign(eps, xmax)

    for i in range(1, maxinter):               # loop over iterations
        xmins = floor(xmin * factor * corrmin)      # smallest integer
        xmaxs = ceil(xmax * factor * corrmax)     # largest integer
        xnint = abs(xmaxs - xmins)        # number of subintervals
        if (xnint <= maxint): break # break if number of subintervals is small enough
        modi = i % 3
        factor = factor * xfact[modi]

    factor = 1e0/factor                       # scale factor
    xmin = xmins * factor                #xmin and xmax
    xmax = xmaxs * factor
    scale = max(Magn(xmin), Magn(xmax))        #Scale factor
    factor = max(abs(xmins), abs(xmaxs))
    for i in range(1, modi + 1): factor = factor / xfact[i]
    nsigd = int(log10(factor) + 1)      #Number of significant digits
    nintv = Nint(xnint)             #Number of subintervals



    return (xmin, xmax, scale, nsigd, nintv)



#======================================================================

def FormStr(x,scale, nsigh):

    """
    Formats the number x (with factor scale) to nsigd significant digits
    Returning the mantissa in mant[] and the exponent of 10 in expn[]

    Returns mant, expn
    """

    ndigmax = 5                                 #maximum number of digits
    mant = expn = ""

    n = Nint(log10(scale))

    if ((n < -1) or (n>3)):
        expn = repr(n)
        x /= scale
        n=0

    n+=1                                  #Number of digits before decimal point
    ndig = min(ndigmax, max(nsigh,n))
    ndec = ndig-n
    x = round(x, ndec)
    mant = "{0:{1:}.{2:}f}".format(x, ndig, ndec)

    return (mant, expn)

def Plot(win,x, y, n, col = "blue", sty = 1,
         fxmin = 0.15, fxmax = 0.95, fymin = 0.15, fymax = 0.90,
         xtext = "x", ytext = "y", title = None):
    """
    :param x: X coordinates, simple list
    :param y: Y coordinates, simple list
    :param n: number of tabulation points
    :param col: plot color
    :param sty: plot style; 0 - scatter plot; 1 -- line plot
    :param fxmin: min fractional x-min (0< fxmin<fxmax<1)
    :param fxmax:
    :param fymin: min fractional y limit (0<fymin<fymax<1)
    :param fymax:
    :param xtext: x axis label
    :param ytext: y axis label
    """

    maxintx = maxinty = 10

    xmin = min(x); xmax = max(x)
    ymin = min(y); ymax = max(y)

    ixmin = Nint(fxmin * nxwin); iymin = Nint((1 - fymin) * nywin)
    ixmax = Nint(fxmax * nxwin); iymax = Nint((1 - fymax) * nywin)


    win.create_rectangle(ixmin, iymax, ixmax, iymin)     #plot frame

    nfont = min(int((ixmax - ixmin)/60.) + 3, 12)     #font size
    font1 = ("Helvetica", nfont)                    #axis label font
    font2 = ("Helvetica", Nint(1.2 * nfont))      #axis title font
    font3 = ("Helvetica", Nint(1.4 * nfont))


    if ((ixmax - ixmin) < 3 * (iymin - iymax)):
        win.create_text((ixmin+ixmax)/2, iymax-3*nfont, text = title, font=font3)
    else:
        win.create_text(ixmax, iymax, text=title, font = font2, anchor = "ne")
        maxinty = max(5, (iymin-iymax) / (ixmax-ixmin) * maxintx)

#----------------------------------------------------------------X-AXIS

    (xmin, xmax, scale, nsigd, nintv) = Limits(xmin, xmax , maxintx)
    ax = (ixmax - ixmin) / (xmax - xmin)   #Scalling coefficients
    bx = ixmin - ax * xmin

    tic = min(ixmax - ixmin, iymin - iymax) / 100.
    h = (xmax - xmin) / nintv; htic = ax * h
    iytext = iymin + 1.5 * nfont
    for i in range(0, nintv + 1):
        ix = Nint(ixmin + i * htic)
        win.create_line(ix, iymin, ix, iymin - tic)
        win.create_line(ix, iymax, ix, iymax + tic)
        if (xtext != "None"):
            mant, expn = FormStr(xmin + i * h, scale, nsigd)
            win.create_text(ix, iytext, text = mant, font = font1)

    if (xtext != "None"):
        if (scale < 0.1 or scale > 1000.): xtext = xtext + " 1e" + expn
        ixtext = (ixmin + ixmax) / 2
        iytext = iytext + 2 * nfont
        win.create_text(ixtext, iytext, text = xtext, font = font2)

#----------------------------------------------------------------Y-AXIS

    if (ymin == 0. and ymax ==0.): ymin =-1.; ymax = 1.
    if (abs(ymax - ymin) < 1e-5 * abs(ymax)): ymin *= 0.9; ymax *= 1.1

    (ymin, ymax ,scale, nsigd, nintv) = Limits(ymin, ymax , maxinty)
    ay = (iymax - iymin) / (ymax - ymin)
    by = iymin - ay * ymin

    h = (ymax - ymin) / nintv; htic = ay * h
    ixtext = ixmin - nfont
    for i in range(0, nintv + 1):
        iy = Nint(iymin + i * htic)
        win.create_line(ixmin, iy, ixmin + tic, iy)
        win.create_line(ixmax, iy, ixmax - tic, iy)
        if (ytext != "None"):
            (mant, expn) = FormStr(ymin + i*h, scale, nsigd)
            win.create_text(ixtext, iy, text = mant, font = font1, anchor = "e")
            lenmant = len(mant)

    if (ytext != "None"):
        if (scale < 0.1 or scale > 1000.): ytext = ytext + " 1e" + expn

        ixtext = ixtext - 3 * nfont / 4 * (len(mant) + 2)
        iytext = (iymax + iymin) / 2

        win.create_text(ixtext, iytext, text=ytext, font=font2, anchor = "e")

    if (ymin * ymax < 0): win.create_line(ixmin, Nint(by), ixmax, Nint(by))
    if (xmin * xmax < 0): win.create_line(Nint(bx), iymin, Nint(bx), iymax)

# -----------------------------------------------------------------PLOT VALUES

    tic = 2 * tic / 3
    if (sty == 4): hx = 0.5 * ax * (x[2] - x[1]) - 1  # half-spacing for histogram
    ix0 = Nint(ax * x[1] + bx)
    iy0 = Nint(ay * y[1] + by)  # 1st point
    for i in range(1, n + 1):
        ix = Nint(ax * x[i] + bx)
        iy = Nint(ay * y[i] + by)  # new point
        if (sty == 0):  # scatter plot
            win.create_rectangle(ix - tic, iy - tic, ix + tic, iy + tic,
                                 outline=col)
        if (sty == 1 or sty == 2):  # line or polar plot
            win.create_line(ix0, iy0, ix, iy, fill=col)
        if (sty == 3):  # drop lines
            win.create_line(ix, by, ix, iy, fill=col)
        if (sty == 4):  # histogram
            win.create_rectangle(ix - hx, iy, ix + hx, by, outline=col)
        ix0 = ix
        iy0 = iy

#==============================================================================

def Plot0(win, x, y, n, col = "blue",
          fxmin = 0.15, fxmax = 0.95, fymin = 0.15, fymax = 0.90,
          xtext = "x", ytext = "y", title = None):
#-------------------------------------------------------------------------------
#  Plots a real function of one variable specified by a set of (x, y) points.
#
#  x[]   - abscissas of tabulation points (x[1] through x[n])
#  y[]   - ordinates of tabulation points (y[1] through y[n])
#  n     - number of tabulation points
#  col   - plot color ("red", "green", "blue" etc.)
#  fxmin - min fractional x-limit of viewport (0 < fxmin < fxmax < 1)
#  fxmax - max fractional x-limit of viewport
#  fymin - min fractional y-limit of viewport (0 < fymin < fymax < 1)
#  fymax - max fractional y-limit of viewport
#  xtext - x-axis title
#  ytext - y-axis title
#  title - plot title
#-------------------------------------------------------------------------------


   xmin = min(x); xmax = max(x)             # user domain limits
   ymin = min(y); ymax = max(y)
                                              # corrections for horizontal plots
   if (ymin == 0.0 and ymax == 0.0): ymin = -1e0; ymax = 1e0
   if (ymin == ymax): ymin *= 0.9; ymax *= 1.1
                                                               # viewport limits
   ixmin = Nint(fxmin * nxwin); iymin = Nint((1.0-fymin) * nywin)
   ixmax = Nint(fxmax * nxwin); iymax = Nint((1.0-fymax) * nywin)
   win.create_rectangle(ixmin, iymax, ixmax, iymin)                 # draw frame
                                                    # axis labels and plot title
   win.create_text((ixmin+ixmax)/2, iymin+10, text = xtext, anchor = N)
   win.create_text(ixmin-10, (iymin+iymax)/2, text = ytext, anchor = E)
   win.create_text((ixmin + ixmax)/2, iymax - 10, text = title, anchor = S)
                                                               # label axes ends
   win.create_text(ixmin, iymin+10, text = "%5.2f" % xmin, anchor = NW)
   win.create_text(ixmax, iymin+10, text = "%5.2f" % xmax, anchor = NE)
   win.create_text(ixmin-10, iymin, text = "%5.2f" % ymin, anchor = E)
   win.create_text(ixmin-10, iymax, text = "%5.2f" % ymax, anchor = E)
                                                         #  scaling coefficients
   ax = (ixmax - ixmin) / (xmax - xmin)                                 # x-axis
   bx = ixmin - ax * xmin
   ay = (iymax - iymin) / (ymax - ymin)                                 # y-axis
   by = iymin - ay * ymin
                                                                     # draw axes
   if (xmin * xmax < 0): win.create_line(int(bx), iymin, int(bx), iymax)   # y
   if (ymin * ymax < 0): win.create_line(ixmin, int(by), ixmax, int(by))   # x

   ix0 = int(ax * x[1] + bx); iy0 = int(ay * y[1] + by)            # 1st point
   for i in range(2, n + 1):
      ix = int(ax * x[i] + bx); iy = int(ay * y[i] + by)           # new point
      win.create_line(ix0, iy0, ix, iy, fill = col)                  # draw line
      ix0 = ix; iy0 = iy                                            # save point
