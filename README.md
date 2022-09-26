This repo is the Python3 version reimplement of [Inpaint.git](https://github.com/fooble/Inpaint.git), which implemented the algorithm of the paper:
***"Region Filling and Object Removal by Exemplar-Based Image Inpainting"** by A.Criminisi et al.*

For simplifying, code for region selection with mouse and mask generation are not implemented, a binary mask image should be provided instead.

The command to run the program is: 
```
python3 main.py <pathOfInputImage> <pathOfMaskImage>[ <patchHalfWidth>]
```

For example:
```
python3 main.py ../cases/image4.jpg ../cases/mask4.jpg 4
```

I did some little optimize to speed up, especially the time-consuming *computeBestPatch()* function.
As a result the it takes about 7 seconds per iteration, for an about 300 * 300 sized image, and patchHalfWidth = 4 (size=9x9), on my ThinkPad laptop made in 2013.
