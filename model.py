import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters(out_channels), stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        # It applies filters (aka kernels) over an image to extract features like edges, textures, shapes, etc.
        # Input Tensor (batch_size, in_channels, height, width)
        # Output Tensor (batch_size, out_channels, new_height, new_width). There will be out_channels amount of filters in out tensor.
        # In forward pass of Conv2d
        #  - Place the kernel on the top-left corner of the image.
        #  - Multiply element-wise and sum the results
        #  - Slide the kernel across the image, repeating the operation for each region.
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)

        # BatchNorm2d helps stabilize and accelerate training by normalizing activations across the batch.
        self.batchnorm = nn.BatchNorm2d(out_channels)

        # ReLU "kills" all negative values → may cause dead neurons (they never activate again)
        # Where negative_slope is a small slope like 0.1, so you don’t completely block negative values.
        # TODO: Try using Parametric ReLU
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolo(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolo, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.cnn_network = self._create_cnn()
        self.fully_connected = self._create_fully_connected(**kwargs)

    def forward(self, x):
        return self.cnn_network(x)

    # Create CNN network that using CNNBlock and Pooling.
    # Output Tensor of this will send to Fully connected layer after flattening.
    def _create_cnn(self):
        layers = []
        in_channels = self.in_channels

        for cl in self.architecture:
            if type(cl) == tuple:
                layers += [
                    CNNBlock(in_channels, cl[1], kernel_size=cl[0], stride=cl[2], padding=cl[3])
                ]
                # Size of out_channel should match the next CNNBlock in_channels for nn.conv2
                in_channels = cl[1]

            elif type(cl) == str:
                # Max pooling layer
                # - Reduce the dimensions(faster training) of the image while keeping important feature.
                # - Slight shift of image does not affect much.
                # - Reduce overfitting(Less params so more generalized trained weights)
                # TODO: Use other pooling like AvgPooling
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))]

            elif type(cl) == list:
                cnn_block_1 = cl[0]
                cnn_block_2 = cl[1]
                num_repeats = cl[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(in_channels, cnn_block_1[1], kernel_size=cnn_block_1[0], stride=cnn_block_1[2], padding=cnn_block_1[3])
                    ]

                    # Size of out_channel should match the next CNNBlock in_channels for nn.conv2
                    layers += [
                        CNNBlock(cnn_block_1[1], cnn_block_2[1], kernel_size=cnn_block_2[0], stride=cnn_block_2[2], padding=cnn_block_2[3])
                    ]
                    in_channels = cnn_block_2[1]

        return nn.Sequential(*layers)
    
    def _create_fully_connected(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes
                
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # out_channels(1024) * out_width(S) * out_hight(s)
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(496, S * S * (C + B + 5)) # 5 is for box center x, y and box width and height
        )

def test(S=7, B=2, C=20):
    model = Yolo(grid_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448)) # batch_size, image_width, image_height, color_channels
    print(model(x).shape)

test()
