# LS-Shufflle-Network
The network consists of multiple LS-Shuffle blocks and the global depthwise convolution (GDWC).

The LS-Shuffle block merges two Shufflenet-v2 blocks by long-path and short-path residuals. 
Hence, there are four convolution layers, two depthwise convolution layers, an addition operation, a channel split layers, a channel concantenate layers, and a channel shuffle layer.


The global depthwise convolution (GDWC) ia applied to replace the global average pooling operation.
