#  Deeplearning for Steganalysis

***
**Steganography and Steganalysis**  



Steganography is the science to conceal secrect messages in the images though slightly modifying the pixel values. Content-adaptive steganographic schemes tend to embed the messages in complex regions to escape from detection are the most secure method in nowadays. Examples in spatial domain include HUGO, WOW, S-UNIWARD.   
Corresponding to steganography, steganalysis is the art of detecting hidden data in images. Usually, this task is formulated as a binary classification problem to distinguish between cover and stego. 


**LSB steganography cover and stego**  

* 1: cover(left) and stego(right)
 
![tu1](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/coverstego.jpg?raw=true)


* 2: the subtraction result of cover and stego(small payload)
 
![subtraction](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/subtraction.jpg?raw=true)


# deeplearning for steganalysis

***
Different from traditional computer vision task, the goal of image steganalysis is to find embedding operation which may be extremely low noise to the cover. So there's no maxpooling layer in my network which could destory small imformations or features caused by Steganography.


# some results

* 3: The training process，the net begins to converge at 50,000 step（5 epoch） 
![Training process](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/S-UNIWARD0.2.png?raw=true)

* 4: WOW0.5random_CNN training and validation accurcy. It can be seen from the validation loss value that the model is not overfitted. Amazing fitting ability.

[^_^]:
![Training process](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/WOW0.5random_CNN.png?raw=true)

<div align=center><img src="https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/WOW0.5random_CNN.png?raw=true"/></div>

***





