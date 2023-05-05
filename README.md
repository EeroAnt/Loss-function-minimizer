# Loss-function-minimizer
Basics of mathematics in machine learning extra assignment

layermaker.py writes a .m-file that has a neuron network with a first layer, at least 1 middle layer and a last layer.
First it asks for the width of the first layer and the first middle layer. After that it asks, if you'd like to add a middle layer
and if you answer (y)es, it will ask for the width of this new layer. This will repeat, until you have opt not to make another
middle layer. The last layer is automatically included and then the rest of configuration happens in matlab.

When running the .m-file you will be asked for the learning rate, how many uniform slices does the training data consist of, 
the amount of datapoints, where they start from and their spacing, for each slice and the the function that you'd like to approximate.

The script matches the first and the last layers' dimensions with your given training data and does starts to iterate gradient descent
to find the minimum for the loss function. It either runs for a bit and stops with no results or it reaches the set precision and halts.
