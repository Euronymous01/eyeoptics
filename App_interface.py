import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import font as tkfont
from tkinter.ttk import *


class App(tk.Tk):
    """
    Frame object holding all the different pages. Controler of the pages.
    """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.titlefont = tkfont.Font(family="Helvetica", size=18, weight="bold")
        self.geometry("800x600+260+50")
        container = tk.Frame(self) # create a frame container that contains the pages
        container.grid(row=0, column=0, sticky="nsew") # make it fill the whole window

        # self.lens = tk.Frame(container) # create a frame for the lens page
        # self.lens.grid(row=0, column=0, sticky="nsew") # make it fill the whole window
        # self.lens.grid_columnconfigure(0, weight=1)
        # self.lens.grid_rowconfigure(0, weight=1)

        self.listing = {} # store the name and object page
        self.pages = {'Lens', 'Eye', 'Settings'} # store the name and object page
        for p in (WelcomePage, Lens, Eye, Settings): #loop through the classes
            page_name = p.__name__ # get the name of the class
            frame = p(parent = container, controller = self) # create an instance of the class
            # Create the parent-child objects
            frame.grid(row=0, column=0, sticky="nsew")
            self.listing[page_name] = frame  #Stored in dictionary
        self.up_frame('WelcomePage') # start on the welcome page

    def up_frame(self, page_name):
        page = self.listing[page_name]
        page.tkraise() # raise the page to the top

class WelcomePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Welcome to the Lens Calculator", font=controller.titlefont, anchor="center")
        label.pack()



        b_lens = tk.Button(self,  text="Lens",
                           command=lambda: controller.up_frame('Lens') , width=10, height=2, bg="blue",
                           fg="white")
        b_lens.pack()


        b_eye = tk.Button(self, text="Eye", comman = lambda : controller.up_frame('Eye'), width=10, height=2, bg="blue", fg="white")
        b_eye.pack()

        b_settings = tk.Button(self, text="Settings", command=lambda: controller.up_frame('Settings'), width=10, height=2, bg="blue", fg="white")
        b_settings.pack()





class Lens(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # self.lens_button = ttk.Button(self, text="Lens", command=lambda: controller.up_frame('Lens'))
        # self.lens_button.pack()

        label = tk.Label(self, text="Lens", font=controller.titlefont)
        label.place(relx=0.5, rely=0.5, anchor="center")
        label.pack(   )

        b_back = tk.Button(self, text="Back to the Main Page",
                           command=lambda: controller.up_frame('WelcomePage'))
        b_back.place(relx=0.5, rely=0.5, anchor="center")
        b_back.pack()


        #Create a graphic
        def lens_canvas():
            lens_canvas = Canvas(self, width=300, height=300, bg="black")

            lens_canvas.create_oval(100, 100, 200, 200, fill="red")
            lens_canvas.pack()

        lens_canvas()



class Eye(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # self.lens_button = ttk.Button(self, text="Lens", command=lambda: controller.up_frame('Lens'))
        # self.lens_button.pack()

        label = tk.Label(self, text="Eye", font=controller.titlefont)
        label.pack()

        b_back = tk.Button(self, text="Back to the Main Page",
                           command=lambda: controller.up_frame('WelcomePage'))
        b_back.pack()

class Settings(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # self.lens_button = ttk.Button(self, text="Lens", command=lambda: controller.up_frame('Lens'))
        # self.lens_button.pack()

        label = tk.Label(self, text="Settings", font=controller.titlefont)
        label.pack()

        b_back = tk.Button(self, text="Back to the Main Page",
                           command=lambda: controller.up_frame('WelcomePage'))
        b_back.pack()



if __name__ == '__main__':
    app = App()
    app.mainloop()