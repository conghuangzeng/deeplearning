from nets import Pnet
import  time
from  trains import  Trainer


if __name__ == '__main__':

    pnet = Pnet()
    tarin_p = Trainer(pnet,"pnet0910.pth","pnet0910.pth",r"C:\celeba\12")#训练轮次：3轮，loss：

    tarin_p.trains()




