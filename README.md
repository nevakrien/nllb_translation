# nllb_translation
simple translation package for python supports quntized infrence on gpu and regular infrence on cpu. 
regular gpu infrence seems unpractical at least for hebrew so I am keeping the code there but not devloping it further



getting the best I can from the translation model using some extra tricks
seems 4-bit works perfectly fine so we use that on gpu and then we can get a lot of beams going and its really good.