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
        if input_ids.size(1) > self.context:
            input_ids = input_ids[:, -self.context:]

        for step in range(self.ngram_size, self.context // 2+ 1):                
            #get all previous slices
            cuts=[input_ids[:,i:i+step] for i in range(len(input_ids[0])-1-(step-1),-1,-step)]
            cuts=cuts[:self.count-1] 

            if(len(cuts)!=self.count-1):
                continue

            matching = torch.ones(input_ids.shape[0], dtype=torch.bool,device=input_ids.device)
            for cut in cuts[1:]:
                matching&= (cut==cuts[0]).all(dim=1)

            x=cuts[0][:,1:]
            if x.size(1)!=0:
                matching&= (input_ids[:,-x.shape[1]:]==x).all(dim=1)
                
            scores[matching,cuts[0][matching,-1]]=float("-inf")

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
    ngram_size=4
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
    assert (scores[bad_beams,1]==float("-inf")).all()
    assert (scores[bad_beams,2:]==1).all()
    #print(scores[bad_beams])
    assert (scores[~bad_beams]==1).all()



def test_stop_repeats2():
    count=3
    ngram_size=4
    context=100
    beam = 10

    processor=StopRepeats(count=count,ngram_size=ngram_size,context=context)
    input_ids=torch.stack([torch.arange(0,context+10,dtype=int) for _ in range(beam)])
    #print(input_ids)

    bad_beams=torch.rand(beam)>0.2
    #print(bad_beams)

    bads=torch.concat([torch.arange(4) for _ in range(context)])[:context]
    input_ids[bad_beams,-context:]=bads
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
    assert (scores[bad_beams,bads[-1]]==float("-inf")).all()

    for i in range(scores.shape[-1]):
        if(i !=bads[-1]):
            assert (scores[bad_beams,i]==1).all()
    #print(scores[bad_beams])
    assert (scores[~bad_beams]==1).all()


#I wrote this because I kept seeing repetissions
#I am pretty sure I am losing my sanity slowly after dealing with this for a few hours

def is_integer_tensor(tensor):
    integer_types = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
    return tensor.dtype in integer_types



import numpy as np
def find_most_repeating_pattern(tensor):
    tensor_length = len(tensor)
    max_pattern_length = 0
    max_repeats = 0
    most_repeating_pattern = None
    
    # Search for all possible patterns by their starting position and length
    for start in range(tensor_length):
        for pattern_length in range(1, tensor_length - start + 1):
            repeats = 1
            pattern = tensor[start:start+pattern_length]
            # Look ahead to see if the pattern repeats consecutively
            for next_start in range(start + pattern_length, tensor_length - pattern_length + 1, pattern_length):
                next_pattern = tensor[next_start:next_start+pattern_length]
                if np.array_equal(pattern, next_pattern):
                    repeats += 1
                else:
                    break  # Break if the pattern does not continue consecutively
            
            # Update the most repeating pattern if this pattern repeats more consecutively
            if repeats > max_repeats or (repeats == max_repeats and pattern_length > max_pattern_length):
                max_repeats = repeats
                max_pattern_length = pattern_length
                most_repeating_pattern = pattern
    
    return most_repeating_pattern, max_repeats

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
    assert(scores[0][54505]==float("-inf"))
    
    t=list(scores[0].cpu())
    t.pop(54505)
    assert all(f==1 for f in t)


import torch
import random

def gpt_real_world_test_repeats():
    # Assuming StopRepeats is defined elsewhere and imported correctly
    processor = StopRepeats(count=3, ngram_size=1, context=40)
    
    # Your original tensor 'x'
    x = torch.IntTensor([77636, 65164, 1060, 126481, 2127, 417, 63597, 248315, 13326, 7063, 1589, 248355, 6212, 71349, 72607, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 8582, 2127, 19499, 248260, 164671, 71349, 77051, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 8582, 2127, 19499, 248260, 248075, 77636, 41833, 1060, 126481, 2127, 417, 63597, 248315, 13326, 7063, 1589, 248355, 6212, 71349, 72607, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 248075, 2])[:-4][None, :]
    
    # Generate a batch of random integer tensors
    batch_size = 3  # Including 'x'
    sequence_length = x.shape[1]
    max_int = x.max() + 1 +1000 # Assuming you want to keep the range similar to 'x'
    
    # Generate random sequences
    random_sequences = [torch.randint(high=max_int, size=(1, sequence_length)) for _ in range(batch_size - 1)]
    
    # Insert 'x' at a random position in the batch
    insert_position = random.randint(0, len(random_sequences))
    random_sequences.insert(insert_position, x)
    
    # Combine into a single batch tensor
    batch_tensor = torch.cat(random_sequences, dim=0)
    
    # Apply the processor
    scores = torch.ones([batch_tensor.size(0), max_int])
    scores = processor(batch_tensor, scores)
    
    # Check the condition for 'x' within the batch
    assert(scores[insert_position][54505] == float("-inf")), "The condition for 'x' did not hold true."
    t=list(scores[insert_position].cpu())
    t.pop(54505)
    assert all(f==1 for f in t)

# Remember to define or import StopRepeats before running this test




if __name__=="__main__":
    real_world_test_repeats()
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