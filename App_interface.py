import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import font as tkfont
from tkinter.ttk import *
import math
from Plotfunction import *

import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


def lens_plot_area():
    width = 500
    heigth = 400
    return width, heigth
    ### In mare parte appul va fi folosit in full screen




class App(tk.Tk):
    """
    Frame object holding all the different pages. Controler of the pages.
    """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.titlefont = tkfont.Font(family="Helvetica", size=18, weight="bold")
        self.geometry("1000x600+260+50")
        container = tk.Frame(self) # create a frame container that contains the pages
        container.grid(row=0, column=0, sticky="nsew") # make it fill the whole window




        self.listing = {} # store the name and object page
        self.pages = {'Lens', 'Eye', 'Settings'} # store the name and object page
        for p in (WelcomePage, Lens, Eye, Settings): #loop through the classes
            page_name = p.__name__ # get the name of the class
            frame = p(parent = container, controller = self) # create an instance of the class
            # Create the parent-child objects
            frame.grid(row=0, column=0, sticky="nsew")
            self.listing[page_name] = frame  #Stored in dictionary
        self.up_frame('WelcomePage') # start on the welcome page

        def callable(event):
            x = event.x
            y=event.y
            print('{},{}'.format(x,y))

        self.bind("<Motion>", callable)
        

    def up_frame(self, page_name):
        page = self.listing[page_name]
        page.tkraise() # raise the page to the top


class WelcomePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Welcome to the Lens Project", font=controller.titlefont, anchor="center")
        label.place(relx = 0.5, rely=0.1)



        b_lens = tk.Button(self,  text="Lens",
                           command=lambda: controller.up_frame('Lens') , width=10, height=2, bg="blue",
                           fg="white")
        b_lens.place(relx=0.65, rely=0.3)


        b_eye = tk.Button(self, text="Eye", comman = lambda : controller.up_frame('Eye'), width=10, height=2, bg="blue", fg="white")
        b_eye.place(relx=0.65, rely=0.4)

        b_settings = tk.Button(self, text="Settings", command=lambda: controller.up_frame('Settings'), width=10, height=2, bg="blue", fg="white")
        b_settings.place( relx=0.65, rely=0.5)






class Lens(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Lens", font=controller.titlefont)
        label.grid(sticky=W, pady=4, padx=5)



        b_back = tk.Button(self, text="Back to the Main Page",command=lambda: controller.up_frame('WelcomePage'))
        b_back.grid(sticky=W, pady=4, padx=5, row=1, column=0)

       
        def lens_canvas():

            width_lens, heigth_lens = lens_plot_area()
            
            def plot():
            
                fig = Figure(figsize = (4,5), dpi=100)
                
                y = [i**2 * np.cos(i) for i in range(101)]
                
                plot1 = fig.add_subplot(111)
                plot1.grid()
                plot1.plot(y)
                
                canvas = FigureCanvasTkAgg(fig, self)
                canvas.drew()
                canvas.get_tk_widget().grid(row=2, column=0, sticky = 'nsew)
                
                toolbar = NavigationToolbar2Tk(canvas, self, pack_toolbar=False)
                toolbar.update()
                canvas._tkcanvas.grid()
        
                   
            plot()
        lens_canvas()

        #Create a drop-off-menu for the lens type

        def drop_down_menu():
            self.lens_type = StringVar()
            self.lens_type.set("Lens Type")
            self.lens_type_menu = OptionMenu(self, self.lens_type, "Convergent", "Divergent", "Spheric", "Aspheric")
            self.lens_type_menu.grid(row=2, column=5, sticky=E+W+S+N, padx=5, pady=5)

        drop_down_menu()
class Eye(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Eye", font=controller.titlefont)
        label.grid(row=0, column=0, columnspan=2, sticky="nsew")

        b_back = tk.Button(self, text="Back to the Main Page",
                           command=lambda: controller.up_frame('WelcomePage'))
        b_back.grid(row=1, column=0, sticky="nsew")

class Settings(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="Settings", font=controller.titlefont)
        label.grid(row=0, column=0, columnspan=2, sticky="nsew")

        b_back = tk.Button(self, text="Back to the Main Page",
                           command=lambda: controller.up_frame('WelcomePage'))
        b_back.grid( row=1, column=0, sticky="nsew")



if __name__ == '__main__':
    app = App()
    app.mainloop()
