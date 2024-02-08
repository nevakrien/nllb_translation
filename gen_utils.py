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

    @torch.no_grad()
    def __call__(self, input_ids, scores):#encoder_input_ids
        #print('raw inputs:')
        #print(input_ids)
        if input_ids.size(1) > self.context:
            input_ids = input_ids[:, -self.context:]

        #print(input_ids.shape)
        #print(input_ids)
        for step in range(self.ngram_size+1, self.context // 2+ 1):                
            #get all previous slices
            #cuts=[input_ids[:,i:i+step] for i in range(len(input_ids[0])-1-(step-1),-1,-step)]
            #print(input_ids[:,:-(step-1)].shape)
            cuts=[input_ids[:,:-(step-1)].flip(dims=[1])[:,i*step:(i+1)*step].flip(dims=[1])  for i in range(self.count-1)]
            cuts=[x for x in cuts if x.shape==cuts[0].shape]
            #cuts=cuts[:self.count-1] 

            if(len(cuts)!=self.count-1):
                continue

            #print(f'start step {step}')
            #print(cuts)
            
            matching = torch.ones(input_ids.shape[0], dtype=torch.bool,device=input_ids.device)
            for cut in cuts[1:]:
                matching&= (cut==cuts[0]).all(dim=1)
                #print(matching)

            #x=cuts[0][:,1:] #seems wrong...
            x=cuts[0][:,:-1]
            #print(x)
            #print(input_ids[:,-x.shape[1]:])
            if x.size(1)!=0:
                matching&= (input_ids[:,-x.shape[1]:]==x).all(dim=1)
                #print(matching)

            if matching.any() and cuts[0].shape[1]:
                #print(scores[matching])
                #print(cuts[0][matching].shape)
                #print(cuts[0][matching,-1])
                scores[matching,cuts[0][matching,-1]]=float("-inf")
            #print(cuts[0][matching,-1])
            #if(step==6):
                #print (scores[matching,cuts[0][matching,-1]])

        return scores

class StopRepeatsDebug(LogitsProcessor):
    #stop repeating values of ngram_size or more inside the context 
    #for instance abcabc is repeating twice has an ngram_size of 3 and fits in a context of 6
    def __init__(self, count,ngram_size,context):
        self.call=StopRepeats(count,ngram_size,context)
        self.inputs=[]
        self.scores=[]
    def __call__(self, input_ids, scores):#encoder_input_ids
        ans=self.call(input_ids,scores)
        self.scores.append(ans.cpu().clone())
        self.inputs.append(input_ids.cpu().clone())
        return ans



def test_stop_repeats():
    count=3
    ngram_size=3
    context=12
    beam = 10

    processor=StopRepeats(count=count,ngram_size=ngram_size,context=context)
    input_ids=torch.stack([torch.arange(0,context+10,dtype=int) for _ in range(beam)])
    #print(input_ids)

    bad_beams=torch.rand(beam)>0.2
    print(f"bad beams:{bad_beams}")
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
    print(scores)
    assert (scores[bad_beams,2:]==1).all()
    #print(scores[bad_beams])
    assert (scores[~bad_beams]==1).all()
    assert (scores[bad_beams,1]==float("-inf")).all()



#I wrote this because I kept seeing repetissions
#I am pretty sure I am losing my sanity slowly after dealing with this for a few hours

def is_integer_tensor(tensor):
    integer_types = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
    return tensor.dtype in integer_types




def real_world_test_repeats():
    processor=StopRepeats(count=3,ngram_size=1,context=40)
    x=torch.IntTensor([ 77636,  65164,   1060, 126481,   2127,    417,  63597, 248315,  13326,
          7063,   1589, 248355,   6212,  71349,  72607,    553,   5620,  72811,
         54505,  27499, 248171,    553,   5620,  72811,  54505,  27499, 248171,
           735,   8582,   2127,  19499, 248260, 164671,  71349,  77051,    553,
          5620,  72811,  54505,  27499, 248171,    553,   5620,  72811,  54505,
         27499, 248171,    553,   5620,  72811,  54505,  27499, 248171,    553,
          5620,  72811,  54505,  27499, 248171,    735,   8582,   2127,  19499,
        248260, 248075,  77636,  41833,   1060, 126481,   2127,    417,  63597,
        248315,  13326,   7063,   1589, 248355,   6212,  71349,  72607,    553,
          5620,  72811,  54505,  27499, 248171,    553,   5620,  72811,  54505,
         27499, 248171,    553,   5620,  72811,  54505,  27499, 248171,    735,
          5620,  72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,
         27499, 248171,    735,   5620,  72811,  54505,  27499, 248171,    735,
          5620,  72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,
         27499, 248171,    735,   5620,  72811,  54505,  27499, 248171,    735,
          5620,  72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,
         27499, 248171, 248075,      2])[:-4]

    x=x[None,:]
    #print(x.shape)
    scores=torch.ones([1,x.max()+1])
    scores=processor(x,scores)
    print((scores==1).all())
    #print(scores[0][54505])
    #assert(scores[0][54505]==float("-inf"))
    assert(scores[0][27499]==float("-inf"))
    
    t=list(scores[0].cpu())
    #t.pop(54505)
    t.pop(27499)
    assert all(f==1 for f in t)


import torch
import random

#buged
# def gpt_real_world_test_repeats():
#     # Assuming StopRepeats is defined elsewhere and imported correctly
#     processor = StopRepeats(count=3, ngram_size=1, context=40)
    
#     # Your original tensor 'x'
#     x = torch.IntTensor([77636, 65164, 1060, 126481, 2127, 417, 63597, 248315, 13326, 7063, 1589, 248355, 6212, 71349, 72607, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 8582, 2127, 19499, 248260, 164671, 71349, 77051, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 8582, 2127, 19499, 248260, 248075, 77636, 41833, 1060, 126481, 2127, 417, 63597, 248315, 13326, 7063, 1589, 248355, 6212, 71349, 72607, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 248075, 2])[:-4][None, :]
    
#     # Generate a batch of random integer tensors
#     batch_size = 3  # Including 'x'
#     sequence_length = x.shape[1]
#     max_int = x.max() + 1 +1000 # Assuming you want to keep the range similar to 'x'
    
#     # Generate random sequences
#     random_sequences = [torch.randint(high=max_int, size=(1, sequence_length)) for _ in range(batch_size - 1)]
    
#     # Insert 'x' at a random position in the batch
#     insert_position = random.randint(0, len(random_sequences))
#     random_sequences.insert(insert_position, x)
    
#     # Combine into a single batch tensor
#     batch_tensor = torch.cat(random_sequences, dim=0)
    
#     # Apply the processor
#     scores = torch.ones([batch_tensor.size(0), max_int])
#     scores = processor(batch_tensor, scores)
    
#     # Check the condition for 'x' within the batch
#     assert(scores[insert_position][54505] == float("-inf")), "The condition for 'x' did not hold true."
#     t=list(scores[insert_position].cpu())
#     t.pop(54505)
#     assert all(f==1 for f in t)

# Remember to define or import StopRepeats before running this test




if __name__=="__main__":
    #real_world_test_repeats()
    for _ in range(100):
        print(_)
        #gpt_real_world_test_repeats()
        test_stop_repeats()
        #test_stop_repeats2()

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