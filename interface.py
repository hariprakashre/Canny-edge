
import os
import cv2 as cv
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

from main_file import *

ip_file_path = 'Input  path' 
op_file_path = 'Output path'

        
def clear_screen():
    for widget in root.winfo_children():
        widget.destroy()

def output_file_path():
    global op_file_path
    op_file_path = filedialog.askdirectory()
    button_op['text'] = op_file_path
    print(op_file_path)

def file_upload():
    sigma   = float(ip_entry_1.get())
    t1      = float(ip_entry_2.get())
    t2      = float(ip_entry_3.get())
    
    global ip_file_path, op_file_path
    ip_file_path = filedialog.askopenfilename()
    print("ip_file:", ip_file_path)
    print("op_file:", op_file_path)

    img = cv.imread(ip_file_path)
    clear_screen()
    img_gray, img_smooth, img_op = Canny_detector(img, sigma, t1, t2)
    cv.imwrite(f"{op_file_path}/img_gray.jpeg", img_gray)
    cv.imwrite(f"{op_file_path}/img_smooth.jpeg", img_smooth)
    cv.imwrite(f"{op_file_path}/img_op.jpeg", img_op)

    image_path= os.path.abspath(f"{op_file_path}/img_gray.jpeg")
    img_1=Image.open(image_path)
    img_1= img_1.resize((450,450), Image.ANTIALIAS)
    global op1
    op1 = ImageTk.PhotoImage(img_1)
    image_label_1 = Label(root, image=op1)
    image_label_1.grid(row=0, column=0, padx=10, pady=10)
    text_label_1 = Label(root, text="GrayScaled Output")
    text_label_1.grid(row=1, column=0, padx=10, pady=10)

    image_path= os.path.abspath(f"{op_file_path}/img_smooth.jpeg")
    img_2=Image.open(image_path)
    img_2= img_2.resize((450,450), Image.ANTIALIAS)
    global op2
    op2 = ImageTk.PhotoImage(img_2)
    image_label_2 = Label(root, image=op2)
    image_label_2.grid(row=0, column=1, padx=10, pady=10)
    text_label_2 = Label(root, text="Gaussian Smoothing Output")
    text_label_2.grid(row=1, column=1, padx=10, pady=10)

    image_path= os.path.abspath(f"{op_file_path}/img_op.jpeg")
    img_3=Image.open(image_path)
    img_3= img_3.resize((450,450), Image.ANTIALIAS)
    global op3
    op3 = ImageTk.PhotoImage(img_3)
    image_label_3 = Label(root, image=op3)
    image_label_3.grid(row=0, column=2, padx=10, pady=10)
    text_label_3 = Label(root, text="Final Output")
    text_label_3.grid(row=1, column=2, padx=10, pady=10)

def validate_ip():
    if ((ip_entry_1.get() == '') or (ip_entry_2.get() =='') or (ip_entry_3.get() == '')):
        ip_label_5 = Label(root, text="**Please make sure all the inputs are given.",fg='red', bg='black', font=("Arial", 10))
        ip_label_5.place(x=300, y=520)
    else:
        file_upload()


root = Tk()
root.title("GNR602 Project Group no.- 20 ")
root.config(bg='#000000',height=750, width=1000)

heading_label = tk.Label(root, text="Canny Edge detection", font=("Arial", 40, "bold"), bg='black',fg='white')
heading_label.place(x=220, y=50)

# input parameters
ip_label_1 = Label(root, text="Sigma(\u03C3):",fg='blue', bg='black', font=("Arial", 24))
ip_label_1.place(x=300, y=150)
ip_entry_1 = Entry(root,font=("Arial", 24))
ip_entry_1.place(x=450, y=150, width=200, height=50)

ip_label_2 = Label(root, text="T1      :",fg='blue', bg='black', font=("Arial", 24))
ip_label_2.place(x=300, y=250)
ip_entry_2 = Entry(root,font=("Arial", 24))
ip_entry_2.place(x=450, y=250, width=200, height=50)

ip_label_3 = Label(root, text="T2      :",fg='blue', bg='black', font=("Arial", 24))
ip_label_3.place(x=300, y=350)
ip_entry_3 = Entry(root,font=("Arial", 24))
ip_entry_3.place(x=450, y=350, width=200, height=50)

# Output file path button
ip_label_4 = Label(root, text="Click here to select output file path.",fg='white', bg='red', font=("Arial", 10))
ip_label_4.place(x=300, y=420)
button_op = Button(root, text="Select Output path", activebackground="#2196F3", activeforeground="white", command=output_file_path,font=("Arial", 12),relief=RIDGE)
button_op.place(x=300, y=450, width=350, height=50)


# Upload Button
button = Button(root, text="Select Input Image", activebackground="#2196F3", activeforeground="white",command=validate_ip,font=("Arial", 20),relief=RIDGE)
button.place(x=300, y=550, width=350, height=50)

root.mainloop()