## MNIST dataset

MNIST dataset is a collection of handwritten digits

(thanks to Yann LeCun for the dataset)

Dataset description:

- Format: ubyte (unsigned byte)
- Image size: 28x28
- ubyte size = 8 bits (possible values from 0 to 255)

About python unsigned int:

It is represented by `4 bytes`

## About internal structure of MNIST dataset for images:


It has 4 unsigned integers at the beginning that indicates what is the file content:

`magic, num_images, rows, cols`

As i said before each one is a uint that you can read using the next line:

```python
import struct

magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))

# the above code means: >: big endian, I: unsigned int (16 bits, unsigned int has 4 bytes, then 4 numbers)
```

## About internal structure of MNIST dataset for labels:
It has 2 unsigned integers at the beginning that indicates what is the file content:
`magic, num_labels`
As i said before each one is a uint that you can read using the next line:

```python
import struct
magic, num_labels = struct.unpack(">II", file.read(8))
# the above code means: >: big endian, I: unsigned int (8 bits, unsigned int has 4 bytes, then 2 numbers)
```