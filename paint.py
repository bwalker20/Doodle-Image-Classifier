from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
import io
import os
import subprocess
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tkinter import messagebox
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


categories = ["cup" , "fish", "fork", "ladder", "tree", "airplane", "donut", "face", "house", "saw", "tent", "sun", "moon", "dog", "table"]
model = load_model('./draw_model.h5')

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()


        self.print_cat_button = Button(self.root, text='print categories', command=self.use_print_cat)
        self.print_cat_button.grid(row=0, column=1)
      
        self.save_button = Button(self.root, text='submit', command=self.use_save)
        self.save_button.grid(row=0, column=2)

        self.clear_button = Button(self.root, text='clear', command=self.use_clear)
        self.clear_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=256, height=256)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 16
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_print_cat(self):
      print(categories)
      messagebox.showinfo('Categories', ', '.join(categories))

    def predict_drawing(self, pic):
        pic = pic / 255
        pic.resize(1, 28, 28, 1)
        prediction = model.predict(pic)
        prediction.resize(15)
        prediction = prediction.tolist()
        index = prediction.index(max(prediction))
        print(categories[index])
        print('Confidence: ', (prediction[index]) * 100, '%')

    def use_save(self):
        self.c.update()
        self.c.postscript(file = "saved_canvas.eps")
        img = Image.open("saved_canvas.eps")
        img.save('file_image.png', 'png')
        x = Image.open('file_image.png', 'r')
        os.remove('saved_canvas.eps')
        os.remove('file_image.png')
        x = x.convert('L')
        x = x.resize((28,28))
        y = np.asarray(x.getdata(), dtype = np.uint8)
        y = np.invert(y)
        self.predict_drawing(y)

    def use_clear(self):
        self.c.delete('all')

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        self.line_width = 16
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
