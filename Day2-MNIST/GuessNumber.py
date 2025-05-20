import tkinter as tk
from PIL import Image, ImageDraw
import torch
from torch import nn 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
        transforms.ToTensor()
])

class NeuralNetwork(nn.Module):
    def __init__ (self):
        super().__init__()
        self.flatten = nn.Flatten() #maybe add back later
        self.SequenceLayer = nn.Sequential (
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
         
    def forward(self, x):
        x = self.flatten(x)
        return self.SequenceLayer(x)

window = tk.Tk() #initalize Tkinter
window.title("GuessYourNumber")

canvas = tk.Canvas(window, width = 280, height = 280, bg='black')
canvas.pack()

image = Image.new("L", (280, 280), color=255)
draw = ImageDraw.Draw(image)


def draw_line(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

model = NeuralNetwork()
model = torch.load('number_model_newTransform.pth', weights_only=False)
model.eval()


def GuessNumber(img):
    img = img.resize((28, 28), resample=Image.LANCZOS).convert("L")
    img = Image.eval(img, lambda x: 255 - x)  # Invert colors
    tensor_img = transform(img).unsqueeze(0)  # Shape: [1, 1, 28, 28]


    with torch.no_grad():
        pred = model(tensor_img)
        guess = pred.argmax(1).item()

    result_label.config(text=f"Guess: {guess}")


canvas.bind("<B1-Motion>", draw_line)

def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", (280, 280), color=255)
    draw = ImageDraw.Draw(image)


btn = tk.Button(window, text='Guess Number', command=lambda: GuessNumber(image))
btn.pack()

clear_btn = tk.Button(window, text='Clear', command= clear_canvas)
clear_btn.pack()

result_label = tk.Label(window, text="Guess: ?", font=("Arial", 20), fg="blue")
result_label.pack()

window.mainloop()
