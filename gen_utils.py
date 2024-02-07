import torch
from transformers import LogitsProcessor
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast


# class StopRepeats(LogitsProcessor):
#     #stop repeating values of ngram_size or more inside the context 
#     #for instance abcabc is repeating twice has an ngram_size of 3 and fits in a context of 6
#     def __init__(self, count,ngram_size,context):
#         self.count = count
#         self.ngram_size=ngram_size
#         self.context = context

#     def __call__(self, input_ids, scores, encoder_input_ids=None):
#         if input_ids.size(1) > self.context:
#             input_ids = input_ids[:, -self.context:]
        
#         #rn we set on count 2 for long sequnces we need to actually
#         for step in range(self.ngram_size, self.context // 2+ 1):
#             for bidx, b in enumerate(input_ids):
#                 b=b[-self.context:]
#                 #curent_stuff=b[1-step:]
#                 # pattern=b[1-2*step:1-step]
#                 # if(len(pattern)!=step):
#                 #     pass#break #this is the source of the bugs!!!
                
#                 cuts=[b[i:i+step] for i in range(len(b)-step,0,-step)]
#                 cuts=cuts[:self.count-1]
#                 if(len(cuts)!=self.count-1):
#                     continue

#                 #print(f"beam is {b}")
#                 if all((x==cuts[0]).all() for x in cuts):
#                     #print(f"setting {bidx},{cuts[0][-1]} because it has this sequnce:\n {cuts[0]}")
#                     scores[bidx][cuts[0][-1]]=float("-inf")
#                 #else:
#                     #print("clear")

#         return scores

class StopRepeats(LogitsProcessor):
    #stop repeating values of ngram_size or more inside the context 
    #for instance abcabc is repeating twice has an ngram_size of 3 and fits in a context of 6
    def __init__(self, count,ngram_size,context):
        self.count = count
        self.ngram_size=ngram_size
        self.context = context

    def __call__(self, input_ids, scores):#encoder_input_ids
        if input_ids.size(1) > self.context:
            input_ids = input_ids[:, -self.context:]
        
        for step in range(self.ngram_size, self.context // 2+ 1):                
            
            cuts=[input_ids[:,i:i+step] for i in range(len(input_ids[0])-step,0,-step)]
            cuts=cuts[:self.count-1]
            if(len(cuts)!=self.count-1):
                continue

            matching = torch.ones(input_ids.shape[0], dtype=torch.bool,device=input_ids.device)
            for cut in cuts[1:]:
               matching&= (cut==cuts[0]).all(dim=1)

            scores[matching,cuts[0][matching,-1]]=float("-inf")
               

        return scores

def test_stop_repeats():
    count=3
    ngram_size=4
    context=10
    beam = 10

    processor=StopRepeats(count=count,ngram_size=ngram_size,context=context)
    input_ids=torch.stack([torch.arange(0,context+10,dtype=int) for _ in range(beam)])
    #print(input_ids)

    bad_beams=torch.rand(beam)>0.2
    #print(bad_beams)
    input_ids[bad_beams,-context:]=1
    #print(input_ids)

    scores=torch.ones([beam,input_ids.max()+13])
    #print(scores.shape)
    #print(input_ids.max())
    #print(input_ids)
    ans=processor(input_ids,scores)

    if ans is scores:
        pass
        #print('modifies')
    #print(scores)
    assert (scores[bad_beams,1]==float("-inf")).all()
    assert (scores[bad_beams,2:]==1).all()
    #print(scores[bad_beams])
    assert (scores[~bad_beams]==1).all()

if __name__=="__main__":
    for _ in range(100):
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