# TODO write function that takes verts and camera params as input and goes to bodypart seg
# can then implement this as Keras lambda layer

# TODO first find correspondence between UP-S31 classes and vertex labelling in template.ply
# TODO then find indices of vertices corresponding to each part
# TODO then project vertices onto image plane (PIXELS
# TODO then place gaussian at each index and sum (over H x W grid) for each part
# https://www.tensorflow.org/api_docs/python/tf/distributions/Normal
# TODO sigmoid
# TODO stack to make H x W x C tensor

