import numpy as np

class Shape:
    def __init__(self, size=(10, 10, 3), fill=(0, 255, 255)):
        self.size = size
        self.fill = fill
    def draw(self):
        rect = np.full(self.size, self.fill)
        return rect

class Grid:
    def __init__(self, size=(10, 10), boxes = []):
        self.size = size
        self.boxes = boxes
    def draw(self):
        rows, cols = self.size
        count = 0
        box_num = len(self.boxes)
        grid = np.array([])
        for i in range(rows):
            row_box = np.array([])
            for j in range(cols):
                dbox = self.boxes[count % box_num].draw()
                count += 1
                if row_box.size == 0:
                    row_box = dbox
                else:
                    row_box = np.hstack((row_box, dbox))
            if grid.size == 0:
                grid = row_box
            else:
                grid = np.vstack((grid, row_box))
        return grid