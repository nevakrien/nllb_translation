# import os

# # Make CUDA devices invisible to PyTorch
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import torch

import os 
from os.path import join
import gc




SEP_TOKEN=2 #specific to nllb I just picked it up

@torch.no_grad()
def _translate_text_chunk(text,tgt_text,tokenizer,model,max_new_tokens=2000,num_beams=3):
    # Tokenize and translate the text
    encoded_text = tokenizer(text, return_tensors="pt")
    #manual fix to hf bug 
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]

    tgt_tokens=tokenizer.encode(tgt_text,add_special_tokens=False)
    tgt_tokens=torch.LongTensor([[SEP_TOKEN,tokenizer.lang_code_to_id[tokenizer.tgt_lang]]+tgt_tokens]).to(model.device)

    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}

    generated_tokens=model.generate(**encoded_text, decoder_input_ids=tgt_tokens,max_new_tokens=max_new_tokens,
            penalty_alpha=0.4,num_beams=num_beams).cpu()

    return tokenizer.decode(generated_tokens[0][tgt_tokens.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def _translate_text_chunk_failsafe(text,tgt_text,tokenizer,model,max_new_tokens=2000,gpu_len=1000):
    # Tokenize and translate the text
    encoded_text = tokenizer(text, return_tensors="pt")
    #manual fix to hf bug 
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]

    tgt_tokens=tokenizer.encode(tgt_text,add_special_tokens=False)
    tgt_tokens=torch.LongTensor([[SEP_TOKEN,tokenizer.lang_code_to_id[tokenizer.tgt_lang]]+tgt_tokens]).to(model.device)

    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}

    #gpu part
    gpu_len = min(gpu_len, max_new_tokens+encoded_text['input_ids'].shape[-1]+tgt_tokens.shape[-1])
    

    if(tgt_tokens.shape[1]>gpu_len or model.device=='cpu'):
        generated_tokens=tgt_tokens

    else:
        print('doing gpu')
        gc.collect() #clear lingering gpu memory in time for next alocation
        generated_tokens=model.generate(**encoded_text, decoder_input_ids=tgt_tokens,max_length=gpu_len,
            ).cpu()#penalty_alpha=0.4).cpu()


    #cpu part
    max_new_tokens-=len(generated_tokens[0][tgt_tokens.shape[1]:])

    if max_new_tokens and generated_tokens.shape[-1]>=gpu_len:
        print('doing cpu')
        device=model.device
        #print(generated_tokens[0][-5:])
        generated_tokens=generated_tokens[:,:-1] #removing eos
        
        model.to('cpu')

        generated_tokens = model.generate(**encoded_text, decoder_input_ids=generated_tokens,max_new_tokens=max_new_tokens,
            )   #penalty_alpha=0.4)

        model.to(device)
        print('done cpu')


    # Decode and return the translated text
    return tokenizer.decode(generated_tokens[0][tgt_tokens.shape[1]:], skip_special_tokens=True)

def get_model_and_tokenizer(tgt_lang="heb_Hebr",src_lang="eng_Latn",cuda=True):
    model_name="facebook/nllb-200-3.3B"
    # Initialize the tokenizer and model
    tokenizer = NllbTokenizerFast.from_pretrained(model_name,tgt_lang=tgt_lang,src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#,load_in_4bit=True)
    #print(model.device)
    
    if(cuda):
        model.to('cuda')

    return model,tokenizer

#gpu only sadly... mainly used for the testing enviorment
def get_quantmodel_and_tokenizer(tgt_lang="heb_Hebr",src_lang="eng_Latn"):
    model_name="facebook/nllb-200-3.3B"
    # Initialize the tokenizer and model
    tokenizer = NllbTokenizerFast.from_pretrained(model_name,tgt_lang=tgt_lang,src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,load_in_4bit=True)
    #model.to('cpu')
    print(model.device)

    return model,tokenizer

def translate_text(text,tokenizer,model,num_beams=3):
    ans=''

    prev=''
    for t in text.split('\n\n'):
        # try:
        prev=_translate_text_chunk(t,prev,tokenizer,model,num_beams=num_beams)
        # except torch.cuda.OutOfMemoryError:
        #     model.to('cpu')
        #     prev=_translate_text_chunk(t,prev,tokenizer,model)
        #     model.to('cuda')
        ans+='\n\n'+prev
    return ans

if __name__=="__main__":
    import sqlite3
    #s="תופעת_רשת"

    path=join('data','wikisql.db')

    conn = sqlite3.connect(path)  # Replace 'your_database.db' with your database file path
    cursor = conn.cursor()

    # Execute a query to select English texts
    #cursor.execute("SELECT text FROM texts WHERE id IN (SELECT id FROM main_data WHERE lang = 'en') LIMIT 5")

    cursor.execute("""
        SELECT texts.text
        FROM main_data
        INNER JOIN texts ON main_data.id = texts.id
        WHERE main_data.lang = 'en'
        LIMIT 5
    """)


    # Fetch the results
    english_texts = [x[0] for x in cursor.fetchall()]

    # Close the database connection
    conn.close()

    # # Print the retrieved English texts
    # for text in english_texts:
    #     print(text)  
    text=english_texts[0]
    
    model,tokenizer=get_quantmodel_and_tokenizer()#get_model_and_tokenizer()
    print(text)
    print(10*'\n')
    trans=translate_text(text,tokenizer,model)
    print(trans)
    