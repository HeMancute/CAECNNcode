#  Deeplearning for Steganalysis

**Steganography and Steganalysis**  


***

Steganography is the science to conceal secrect messages in the images though slightly modifying the pixel values. Content-adaptive steganographic schemes tend to embed the messages in complex regions to escape from detection are the most secure method in nowadays. Examples in spatial domain include HUGO, WOW, S-UNIWARD.   
Corresponding to steganography, steganalysis is the art of detecting hidden data in images. Usually, this task is formulated as a binary classification problem to distinguish between cover and stego. 


# LSB steganography cover and stego



* 1: cover(left) and stego(right)
 
![tu1](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/coverstego.jpg?raw=true)


* 2: the subtraction result of cover and stego(small payload)
 
![subtraction](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/subtraction.jpg?raw=true)


# deeplearning for steganalysis(some results)
* 3: The training process，the net begins to converge at 50,000 step 
![Training process](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/S-UNIWARD0.2.png?raw=true)

* 4: WOW0.5random_CNN training and validation accurcy 
![Training process](https://github.com/jiangszzzzz/CAECNNcode/blob/master/data/WOW0.5random_CNN.png?raw=true)





