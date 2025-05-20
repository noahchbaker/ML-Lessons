#You can save and load this model using torch.save(model.state_dict(), 'model.pth') and model.load_state_dict(...).
#The drawing needs to be resized to 28x28 (MNIST size) and converted into a tensor.

import tkinter as tk
from PIL import Image, ImageDraw

# Setup Tkinter window
root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=280, height=280, bg='black')
canvas.pack()

# Image for PIL to store the drawing
image = Image.new("L", (280, 280), color=255)
draw = ImageDraw.Draw(image)

# Draw using mouse drag
def draw_line(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='white')
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

canvas.bind("<B1-Motion>", draw_line)
root.mainloop()
