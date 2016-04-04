# uNeural - Fixed point neural network library suitable for microcontrollers

Ever want to run a feedforward neural network on a microcontroller?
Cool, me too.

This library is intended to provide a generic way to construct, train,
and implement fully connected feedforward artificial neural networks.
It makes use of the excellent libfixmath library to ensure reasonable
performance on low end cores without the use of a floating point math
coprocessor.  Also of note: where appropriate, all fixed point math
ops use their saturating variants, to ensure overflow never occurs.

## Building

To build libneural.a, simply type `make`.  If you are cross compiling
(for example, targeting a microcontroller), you may provide a cross
compiler argument in the form `make CROSS=arm-none-eabi-` (note the
inclusion of the trailing dash).

## Examples

Examples can be built via `make example`.

### Simple network

A basic 4 input, single output feedforward network with a single, 3
element hidden layer.  Training parameters and layer type can be
modified by changing the defintions at the top of the file.
