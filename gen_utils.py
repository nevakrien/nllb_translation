import torch
from transformers import LogitsProcessor
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

class StopRepeats(LogitsProcessor):
    #stop repeating values of ngram_size or more inside the context 
    #for instance abcabc is repeating twice has an ngram_size of 3 and fits in a context of 6
    def __init__(self, count,ngram_size,context):
        self.count = count
        self.ngram_size=ngram_size
        self.context = context

    def __call__(self, input_ids, scores, encoder_input_ids=None):
        s=tuple(scores.shape)
        if input_ids.size(1) > self.context:
            input_ids = input_ids[:, -self.context:]
        

        for step in range(self.ngram_size, self.context // 2):
            patterns=input_ids[:,1-2*step:1-step] #the pattern that WOULD be repeated
            #print(patterns)
            #print(10*"\n")
            #print(input_ids)
            if(patterns.shape[-1]==0):
                continue
            matching = torch.ones(input_ids.shape[0], dtype=torch.bool,device=input_ids.device)
            #print(matching.shape)
            
            #BUG: this loop is wrong
            for i in range(2, self.count):
                x=input_ids[:,1-step*(i+1):1-step*i].shape
                # if(input_ids.shape[0]+1-step*(i+1)<0):
                #     #print(f"index {i} is problematic")
                #     continue
                # print("yay")
                #print(input_ids[:,1-step*(i+1):1-step*i].shape)
                matching &=(patterns==input_ids[:,1-step*(i+1):1-step*i]).all(dim=1)

            if matching.all():
                continue

            #print(scores[~matching].shape)
            scores[~matching,patterns[:,-1]]=float("-inf") #dosent work
            assert scores.shape==s
            #patterns=patterns[:,1]
            #patterns
        
        assert scores.shape==s
        return scores


def test_stop_repeats():
    count=3
    ngram_size=3
    context=100
    beam = 2

    processor=StopRepeats(count=count,ngram_size=ngram_size,context=context)
    input_ids=torch.stack([torch.arange(0,context,dtype=int) for _ in range(beam)])
    #print(input_ids)

    bad_beams=torch.rand(beam)>0.2
    print(bad_beams)
    input_ids[bad_beams,-context:]=1
    #print(input_ids)

    scores=torch.ones([beam,input_ids.max()])
    ans=processor(input_ids,scores)

    if ans is scores:
        pass
        #print('modifies')
    print(scores)
    assert (scores[~bad_beams]==1).all()
    assert (scores[bad_beams,1]==float("-inf")).all()
    assert (scores[bad_beams,2:]==1)

if __name__=="__main__":
    test_stop_repeats()

    model_name="facebook/nllb-200-3.3B"
    # Initialize the tokenizer and model
    tokenizer = NllbTokenizerFast.from_pretrained(model_name)
    inv_vocab={v:k for k,v in tokenizer.vocab.items()}

    x=tokenizer.encode('hey \n\n there \n\n')
    #print(dir(tokenizer))
    print(x)
    #print(tokenizer.encode('hey \n\n there'))
    print([inv_vocab[t] for t in x])

    x=tokenizer.encode('hey there \n\n')
    #print(dir(tokenizer))
    print(x)
    #print(tokenizer.encode('hey \n\n there'))
    print([inv_vocab[t] for t in x])