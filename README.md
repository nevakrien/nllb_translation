# nllb_translation
simple translation package for python supports quntized infrence on gpu and regular infrence on cpu. 
regular gpu infrence seems unpractical at least for hebrew so I am keeping the code there but not devloping it further



getting the best I can from the translation model using some extra tricks
seems 4-bit works perfectly fine so we use that on gpu and then we can get a lot of beams going and its really good.

## env 
get stanza (it will attempt to overwrite ur pytorch installation...)
after it is installed uninstall pytorch cpu and install pytorch gpu again.

for xpu on intel cloud its best to use python 9 with this
python -m pip install torch==1.13.0a0+git6c9b55e torchvision==0.14.1a0 intel-extension-for-pytorch==1.13.120+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu-idp/us/
