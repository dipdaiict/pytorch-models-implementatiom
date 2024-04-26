# Deep Learning Model Implementation in PyTorch

## Calculations
### Calculation of Output Size
The formula to calculate the output size at each layer is given by:
```
Output Size = ((Height/Width + 2 * Padding - Kernel Size) / Strides) + 1
```
Where:
- `Height/Width`: Input height or width (depending on the dimension being convolved).
- `Padding`: Number of zero-padding pixels.
- `Kernel Size`: Size of the convolutional kernel/filter.
- `Strides`: Stride of the convolution operation.

### Calculation of Parameters
The formula to calculate the parameters at each layer is given by:
```
Parameters = (((Kernel Size * Kernel Size) * Number of Channels) + 1) * Number of Filters
```
Where:
- `Kernel Size`: Size of the convolutional kernel/filter.
- `1`: Number of bias parameters (usually 1 per filter).
- `Number of Filters`: Number of filters in the convolutional layer.
- `Number of Channels`: Number of channels of the Images or Feature Maps

Activation Layer has No Parameters.

