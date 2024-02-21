from translate import get_model_and_tokenizer
#from optimum.intel import INCQuantizer
from optimum.intel import OVQuantizer
from neural_compressor.config import PostTrainingQuantConfig

#def quntize_model(model,save_dir='quntized_model'):
#    #activly lying to the lib to force it to quntize
#    quantization_config = PostTrainingQuantConfig(approach="dynamic")
#    quantizer = INCQuantizer.from_pretrained(model)
#    quantizer.quantize(
#        quantization_config=quantization_config,
#        save_directory=save_dir,
#    )

def quntize_model(model,save_dir='quntized_model'):
     quantizer=OVQuantizer.from_pretrained(model)
     quantizer.quantize(save_directory=save_dir, weights_only=True)


if __name__=="__main__":
        model,tokenzier=get_model_and_tokenizer()
        quntize_model(model)
