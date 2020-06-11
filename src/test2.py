# imports every file form tkinter and tkinter.ttk
import tkinter as tk


def motion(event):
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))

def keypress(event):
    """The 4 key press"""
    x, y = 0, 0
    if event.char == "a":
        x = -10
    if event.char == "d":
        x = 10
    if event.char == "w":
        y = -10
    if event.char == "s":
        y = 10
    canvas.move(r1, x, y)
    #canvas.move(t, x, y)


root = tk.Tk()

canvas_width = 600
canvas_height = 400
w = canvas_width // 2
h = canvas_height // 2
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

r1 = canvas.create_rectangle(w, h, w + 40, h + 10)
#t = canvas.create_text(w + 20, h + 5, text="Hello")

#root.bind('<Motion>', motion)
root.bind("<Key>", keypress)
root.mainloop()
