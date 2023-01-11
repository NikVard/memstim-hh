# BASIC TKINTER CHEATSHEET
# Build basic GUIs with Python

from tkinter import *
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter.ttk import Progressbar
from tkinter import filedialog
from tkinter import Menu

# create blank window
window = Tk()
window.title("Welcome to LikeGeeks app")

# create a label widget with font size
lbl = Label(window, text = "Hello", font = ("Arial Bold",20))
lbl.grid(column = 0, row = 0)

# setting window size
window.geometry('500x500')

# handle button click event
# handles textbox
# has to be placed before the button
def clicked():
    res = "Welcome to " + txt.get()
    lbl.configure(text = res)
    print(selected.get())
    # prints to the terminal

    #MESSAGE BOXES
    #messagebox.showinfo('Message title', 'Message content')

# adding a button widget with colors
# why can't the bg change?
btn = Button(window, text = "Click Me", bg = "black", fg = "black", command = clicked, font = ("Arial",20))
btn.grid(column = 0, row = 2)
# puts it in the second column of the window

# get input using entry class
txt = Entry(window, width = 0)
# use the grid function as usual to add it to the window
txt.grid(column = 0, row = 1)

# set focus to entry widget -> can write text right away
txt.focus()

# Disable entry widget
# -- txt = Entry(window, width=10, state='disabled')

# Add a combobox widget
# -- combo = Combobox(window)
# -- combo['values'] = (1,2,3,4,5, "Text")
# -- combo.current(1)
# -- combo.grid(column = 0, row = 3)

# Add a Checkbutton widget
chk_state = BooleanVar()
chk_state.set(True)
chk = Checkbutton(window, text = 'Choose', var = chk_state)
chk.grid(column = 0, row = 4)
# set the checked state (var=_chk_state) by passing the checkvalue to the checkbutton
# use IntVar to set value to 0 or 1 instead of BooleanVar

# Add radio buttons widgets
selected = IntVar()
rad1 = Radiobutton(window,text='First', value=1, variable=selected)
rad2 = Radiobutton(window,text='Second', value=2, variable=selected)
rad3 = Radiobutton(window,text='Third', value=3, variable=selected)

rad1.grid(column = 0, row = 5)
rad2.grid(column = 1, row = 5)
rad3.grid(column = 2, row = 5)

# ScrolledText widget
txt = scrolledtext.ScrolledText(window, width = 40, height = 10)
txt.grid(column = 0, row = 6)
# set scrolledtext content
txt.insert(INSERT, 'Your text goes here')
# delete/clear scrolledtext content
# -- txt.delete(1.0, END)

# Create a MessageBox
# -- messagebox.showinfo('Message title', 'Message content')
# -- messagebox.askquestion('Message title', 'Message content') # yes no
# -- messagebox.askyesno('Message title', 'Message content')
# -- messagebox.askokcancel('Message title', 'Message content')
# -- messagebox.askretrycancel('Message title', 'Message content')
messagebox.askyesnocancel('Message title', 'Message content')
# ok, yes, retry returns TRUE
# no, cancel returns FALSE

# Add a spinbox (numbers widget)
spin = Spinbox(window, from_ = 0, to = 100, width = 10)
spin.grid(column = 0, row = 7)
# you can specify the numbers for the Spinbox
spin3 = Spinbox(window, values = (3,8,11), width = 5)
spin.grid(column = 2, row = 7)

# Set the value for Spinbox
var = IntVar()
var.set(36)
spin2 = Spinbox(window, from_=0, to=100, width=5, textvariable=var)
spin2.grid(column = 1, row = 7)

# Add a progressbar widget, and change the color
# -- style = ttk.Style()
# -- style.theme_use('default')
# -- style.configure("black.Horizontal.TProgressbar", background = 'black')
# -- bar = Progressbar(window, length = 200, style = 'black.Horizontal.TProgressbar')
bar = Progressbar(window, length = 200)
bar['value'] = 70
bar.grid(column = 0, row = 8)

# Add a file dialog (file and directory chooser)
def openfile():
    filedialog.askopenfilenames()
# -- file = filedialog.askopenfilename()
# -- files = filedialog.askopenfilenames() # ask for multiple files
openfiles = Button(window, text = "OpenFiles", bg = "black", fg = "black", command = openfile, font = ("Arial",20))
openfiles.grid(column = 0, row = 9)

# Specify file types (filter file extensions)
def openfile2():
    filedialog.askopenfilename(filetypes = (("Text files","*.txt"),("all files","*.*")))
openfiles2 = Button(window, text = "OpenFiles2", bg = "black", fg = "black", command = openfile2, font = ("Arial",20))
openfiles2.grid(column = 0, row = 10)

# ask for a directory
def openfile3():
    filedialog.askdirectory()
openfiles3 = Button(window, text = "Ask Directory", bg = "black", fg = "black", command = openfile3, font = ("Arial",20))
openfiles3.grid(column = 0, row = 11)

# specify initial directory for the file dialog by specifying initaldir
# from os import path
# -- file = filedialog.askopenfilename(initialdir= path.dirname(__file__))

# Add a menu bar -> NOT WORKING
menu = Menu(window)
new_item = Menu(menu)
new_item.add_command(label='New', command = clicked)
menu.add_cascade(label='File', menu=new_item)
window.config(menu=menu)

# Add Notebook widget (tab) --> uses ttk
# -- tab_control = ttk.Notebook(window)
# -- tab1 = ttk.Frame(tab_control)
# -- tab_control.add(tab1, text='First')
# -- tab_control.pack(expand=1, fill='both')

# Add widgets to Notebook
# -- tab_control = ttk.Notebook(window)
# -- tab1 = ttk.Frame(tab_control)
# -- tab2 = ttk.Frame(tab_control)
# -- tab_control.add(tab1, text='First')
# -- tab_control.add(tab2, text='Second')
# -- lbl1 = Label(tab1, text= 'label1')
# -- lbl1.grid(column=0, row=0)
# -- lbl2 = Label(tab2, text= 'label2')
# -- lbl2.grid(column=0, row=0)
# -- tab_control.pack(expand=1, fill='both')

# Add spacing to widgets
# -- lbl1 = Label(tab1, text= 'label1', padx=5, pady=5)
# using padx and pady

# PROBABLY THE MOST IMPORTANT PART!!!!!!!!!!

# if you forget to call the mainloop function, nothing will appear to the user
window.mainloop()
