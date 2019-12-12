import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import  numpy as np
# m = nn.Sigmoid()
x = torch.arange(1,10).float()

y= F.softmax(x)
plt.plot(x,y)
plt.ion()
# plt.show()
plt.pause(5)
plt.ioff()