from nets import Rnet

from  trains import  Trainer


if __name__ == '__main__':
    rnet = Rnet()
    tarin_r = Trainer(rnet,"rnet0920.pth","rnet0924.pth",r"C:\celeba\24")

    tarin_r.trains()