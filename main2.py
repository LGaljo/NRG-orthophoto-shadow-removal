# Thanks to Basj from StackOverflow for an example of advanced image viewer
# https://stackoverflow.com/a/48137257/16927038

# -*- coding: utf-8 -*-
# Advanced zoom for images of various types from small to huge up to several GB
import math
import warnings
import tkinter as tk
from tkinter import filedialog

from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageTk
from PIL.Image import Resampling

from st_cgan.main import ST_CGAN
from tools.fetch_images import AtlasOkoljaFetcher


class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)


class CanvasImage:
    """ Display and zoom image """

    def __init__(self, placeholder, path):
        """ Initialize the ImageFrame """
        self.manipulations = []
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Resampling.LANCZOS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramid scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def load_image(self, path):
        self.path = path
        # Decide if this image huge or not
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramid scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def grid_remove(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid_remove()  # place CanvasImage widget on the grid

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.manipulations.append(('scroll_x', *args))
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.manipulations.append(('scroll_y', *args))
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)
        self.manipulations.append(('move_from', event))

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.manipulations.append(('move_to', event))
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        self.manipulations.append(('wheel', event))
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        # if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def repeat_manipulation(self, manipulations):
        for action in manipulations:
            # print(action[0])
            if action[0] == 'move_from':
                self.__move_from(action[1])
            if action[0] == 'move_to':
                self.__move_to(action[1])
            if action[0] == 'wheel':
                self.__wheel(action[1])
            if action[0] == 'scroll_x':
                self.__scroll_x(action[1])
            if action[0] == 'scroll_y':
                self.__scroll_y(action[1])

    def clear_manipulations(self):
        self.manipulations = []

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()


class MainWindow(ttk.Frame):
    """ Main window class """

    def __init__(self, mainframe, path):
        self.st_cgan = None
        self.img = True
        self.progressbar = None
        self.loading_bar = None
        self.lid = "lay_ao_dof_2019"
        self.tile_size = (20, 20)
        self.entry1 = None
        self.entry2 = None
        self.entry3 = None
        self.entry4 = None

        self.path = path
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Image Viewer')
        self.master.geometry('1200x900')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        # Grid for image A
        self.canvasA = CanvasImage(self.master, self.path)
        self.canvasA.grid(row=0, column=0, columnspan=4, sticky="nsew")

        # Grid for image B
        self.canvasB = CanvasImage(self.master, self.path)
        self.canvasB.grid(row=0, column=0, columnspan=4, sticky="nsew")

        self.convert_button = ttk.Button(self.master, text="Remove shadows", command=self.remove_shadows)
        self.convert_button.grid(row=1, column=0, padx=10, pady=10)

        self.compare_button = ttk.Button(self.master, text="Compare with other", command=self.compare_images)
        self.compare_button.grid(row=1, column=1, padx=10, pady=10)
        self.compare_button.grid_remove()

        self.load_button = ttk.Button(self.master, text="Load Image", command=self.browse_image)
        self.load_button.grid(row=1, column=2, padx=10, pady=10)

        self.fetch_button = ttk.Button(self.master, text="Fetch image", command=self.show_download_window)
        self.fetch_button.grid(row=1, column=3, padx=10, pady=10)

    def browse_image(self):
        """ Open a file dialog to select an image file """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            print('load image')
            self.path = file_path
            self.canvasA = CanvasImage(self.master, file_path)
            self.canvasA.grid(row=0, column=0, columnspan=4, sticky="nsew")

    def remove_shadows(self):
        self.st_cgan = ST_CGAN("./st_cgan/model/ST-CGAN_G1.pth", "./st_cgan/model/ST-CGAN_G2.pth")

        image = Image.open(self.path)
        image_tensor = torch.tensor(np.array(image))
        tiles = image_tensor.unfold(0, 256, 256).unfold(1, 256, 256)
        rows = tiles.shape[0]
        cols = tiles.shape[1]
        # merged_image = torch.zeros((rows * 256, cols * 256, image_tensor.shape[2]), dtype=torch.uint8)
        merged_image2 = Image.new("RGB", (rows * 256, cols * 256))

        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                start_row = i * 256
                start_col = j * 256
                tile = tiles[i, j]
                reshaped_image = tile.permute(1, 2, 0).contiguous().view(256, 256, 3)
                tile_image = Image.fromarray(reshaped_image.numpy().astype(np.uint8), 'RGB')
                tile_image, _ = self.st_cgan.convert_image(tile_image)

                # Calculate the position of the tile in the merged image
                paste_position = (start_col, start_row)

                # Paste the tile into the merged image
                merged_image2.paste(tile_image, paste_position)

                # merged_image[start_row:start_row + 256, start_col:start_col + 256, :] = torch.tensor(np.array(tile_image))

        # merged_image_pil = Image.fromarray(merged_image.numpy())
        out_path = self.path[:-4] + "_shadowless" + self.path[-4:]
        merged_image2.save(out_path)

        print(f"Image saved to {out_path}")

        # for x in range(start_r, end_r):
        #     for y in range(start_c, end_c):
        #         in_path = Image.open("./img/%s_%s_%s_%s.jpg" % (self.lid, str(x), str(y), str(self.entry3.get())))
        #         out_path = Image.open("./out/%s_%s_%s_%s_shadowless.jpg" % (self.lid, str(x), str(y), str(self.entry3.get())))
        #         image_to_process.append(in_path)
        #         image_out.append(out_path)

        self.canvasA.clear_manipulations()
        self.canvasB.load_image(out_path)
        self.canvasA.grid_remove()
        self.canvasB = CanvasImage(self.master, out_path)
        self.canvasB.grid(row=0, column=0, columnspan=4, sticky="nsew")

        self.compare_button.grid()

    def compare_images(self):
        if self.img:
            self.canvasA.grid_remove()
            self.canvasB.repeat_manipulation(self.canvasA.manipulations)
            self.canvasA.clear_manipulations()
            self.canvasB.clear_manipulations()
            self.canvasB.grid()
        else:
            self.canvasB.grid_remove()
            self.canvasA.repeat_manipulation(self.canvasB.manipulations)
            self.canvasA.clear_manipulations()
            self.canvasB.clear_manipulations()
            self.canvasA.grid()
        self.img = not self.img

    def show_download_window(self):
        child_w = tk.Toplevel(root)
        child_w.geometry("200x150")
        child_w.title("Fetch images")
        child_w.transient(root)
        child_w.grab_set()
        child_w.lift()

        # First input field and label
        label1 = tk.Label(child_w, text="Row:")
        label1.grid(row=0, column=0, sticky="e")
        self.entry1 = tk.Entry(child_w)
        self.entry1.grid(row=0, column=1)

        # Second input field and label
        label2 = tk.Label(child_w, text="Column:")
        label2.grid(row=1, column=0, sticky="e")
        self.entry2 = tk.Entry(child_w)
        self.entry2.grid(row=1, column=1)

        # Third input field and label
        label3 = tk.Label(child_w, text="LOD:")
        label3.grid(row=2, column=0, sticky="e")
        self.entry3 = tk.Entry(child_w)
        self.entry3.grid(row=2, column=1)
        self.entry3.insert(0, str(16))

        # Fourth input field and label
        label4 = tk.Label(child_w, text="Width:")
        label4.grid(row=3, column=0, sticky="e")
        self.entry4 = tk.Entry(child_w)
        self.entry4.grid(row=3, column=1)
        self.entry4.insert(0, str(20))

        fetch_button = ttk.Button(child_w, text="Fetch images", command=self.download)
        fetch_button.grid(row=4, column=0, pady=10, columnspan=2)

    def download(self):
        if self.entry1.get() != "" and self.entry2.get() != "" and self.entry3.get() != "" and self.entry4.get() != "":
            params = {
                'r': int(self.entry1.get()),
                'c': int(self.entry2.get()),
                'lod': int(self.entry3.get()),
                'f': 'jpg',
                'lid': 'lay_ao_dof_2019',
                'gcid': 'lay_AO_DOF_2019',
                'width': int(self.entry4.get()),
                'output': 'img',
            }
            fetcher = AtlasOkoljaFetcher(params)
            fetcher.fetch()
            self.path, _ = fetcher.merge()
            self.canvasA.load_image(self.path)


if __name__ == '__main__':
    filename = 'out/lay_ao_dof_2019_1162_1283_16.jpg'  # place path to your image here
    root = tk.Tk()
    app = MainWindow(root, path=filename)
    app.mainloop()
