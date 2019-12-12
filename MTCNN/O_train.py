from nets import Onet

from  trains import  Trainer


if __name__ == '__main__':
    onet = Onet()
    tarin_o = Trainer(onet,"onet0919.pth","onet0924.pth",r"C:\celeba\48")

    tarin_o.trains()