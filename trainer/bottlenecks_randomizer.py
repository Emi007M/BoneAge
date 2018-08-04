import random
import numpy as np

class BottlenecksRandomizer:
    def __init__(self, category, image_lists):
        self.category = category
        self.image_lists = image_lists
        self.labels = []
        self.labels_offset = [0]
        self.images_amount = 0
        for label_index, label_name in enumerate(image_lists.keys()):
            counter = int(len(image_lists[label_name][category]))
            self.images_amount += counter
            self.labels.append(counter)
            self.labels_offset.append(self.images_amount)

    def choose_random_bottleneck(self):
        index = random.randrange(self.images_amount)
        label_index = np.argmax(np.asarray(self.labels_offset) > index) - 1
        label_name = list(self.image_lists.keys())[label_index]
        image_index = index - self.labels_offset[label_index]
        return label_index, label_name, image_index

    def clean_after_choosing(self, label_index, label_name, image_index):
        #self.print()

        self.images_amount -= 1
        self.labels[label_index] -= 1
        self.labels_offset[label_index + 1:] = map(lambda x: x - 1, self.labels_offset[label_index + 1:])

        del self.image_lists[label_name][self.category][image_index]
        if len(self.image_lists[label_name][self.category]) is 0:
            del self.image_lists[label_name]

            del self.labels[label_index]
            del self.labels_offset[label_index + 1]

    def print(self):
        print("BR: " + self.category + " " + str(self.labels) + " -> " + str(self.images_amount) + " > " + str(self.labels_offset))
