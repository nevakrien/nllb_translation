# nllb_translation
simple translation package for python supports quntized infrence on gpu and regular infrence on cpu. 
regular gpu infrence seems unpractical at least for hebrew so I am keeping the code there but not devloping it further



getting the best I can from the translation model using some extra tricks
seems 4-bit works perfectly fine so we use that on gpu and then we can get a lot of beams going and its really good.

## bugs
the StopRepeats code I made seems to not work for some reason... 
I have tested the cases in which I saw repetisions by checking the wrong tensors manualy.
seems like it does what its supposed to and sets them to -inf so I have NO idea why those tokens get chosen...

seems like spliting the text properly is the most important thing here
thinking how to do that now

