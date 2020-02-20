This is the Python reimplement of [Inpaint.git](https://github.com/fooble/Inpaint.git), which implemented the algorithm of the paper:
***"Region Filling and Object Removal by Exemplar-Based Image Inpainting"** by A.Criminisi et al.*

For simplifying, code for region selection with mouse and mask generation are not implemented.

You could use command: 
```
python main.py pathOfInputImage pathOfMaskImage [halfPatchWidth]
```
to run the program.
For example:
```
python main.py ../tests/image4.jpg ../tests/mask4.jpg 4
```

I did some little optimize to speed up, especially the time-consuming *computeBestPatch()* function.
Now the iteration time is about 7 sec, for image with size about 300 * 300 and halfPatchWidth = 4, on my ThinkPad laptop made in 2013.
