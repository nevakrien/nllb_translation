#from translate import get_quantmodel_and_tokenizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import torch
import os 
from os.path import join
import stanza

SEP_TOKEN=2 #specific to nllb I just picked it up

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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
    #model.to('cpu')
    #print(model.device)

    return model,tokenizer

@torch.no_grad()
def translate_text_chunk(text,tokenizer,model,max_new_tokens=2000,num_beams=3):
    lang_token=tokenizer.lang_code_to_id[tokenizer.tgt_lang]
    # Tokenize and translate the text
    encoded_text = tokenizer(text, return_tensors="pt")
    #manual fix to hf bug 
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]


    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}

    generated_tokens=model.generate(**encoded_text,forced_bos_token_id=lang_token,
    	max_new_tokens=max_new_tokens,num_beams=num_beams,
            no_repeat_ngram_size=7,#conservative about it
            length_penalty=1.2,).cpu()#penalty_alpha=0.4,repetition_penalty=1.2,).cpu()

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

#repetition_penalty
@torch.no_grad()
def _translate_text_chunk_in_context(text,tgt_text,tokenizer,model,max_new_tokens=2000,num_beams=3):
    # Tokenize and translate the text
    encoded_text = tokenizer(text, return_tensors="pt")
    #manual fix to hf bug 
    encoded_text['input_ids'][:,1]=tokenizer.lang_code_to_id[tokenizer.src_lang]

    tgt_tokens=tokenizer.encode(tgt_text,add_special_tokens=False)
    tgt_tokens=torch.LongTensor([[SEP_TOKEN,tokenizer.lang_code_to_id[tokenizer.tgt_lang]]+tgt_tokens]).to(model.device)

    encoded_text={k:v.to(model.device) for k,v in encoded_text.items()}

    generated_tokens=model.generate(**encoded_text, decoder_input_ids=tgt_tokens,max_new_tokens=max_new_tokens,num_beams=num_beams,
			no_repeat_ngram_size=7,#conservative about it
            length_penalty=1.2,).cpu()#penalty_alpha=0.4,repetition_penalty=1.2,).cpu()

    return tokenizer.decode(generated_tokens[0][tgt_tokens.shape[1]:], skip_special_tokens=True)

def chunk_gen(text,tokenizer,spliter):
	doc = spliter(text)

	# Iterate over sentences and print them
	for sentence in doc.sentences:
	    yield (sentence.text)


	# #3 extra tokens for sep lang ... sep
	# toks=tokenizer.encode(text,add_special_tokens=False)
	# while(len(toks)>=(100-3)):
	# 	to_take=toks[:360-3]
	# 	yield tokenizer.decode(to_take, skip_special_tokens=True)
	# 	try:
	# 		toks=toks[len(to_take):]
	# 	except IndexError:
	# 		return

def gen_translate_text_pairs(text,spliter,tokenizer,model,num_beams=10,max_new_tokens=10**4):
    prev=''
    prev_source=''
    for t in text.split('\n\n'):
        for chunk in chunk_gen(t,tokenizer,spliter):
        	#ans=translate_text_chunk(chunk,tokenizer,model,num_beams=num_beams,max_new_tokens=max_new_tokens)
        	input_chunk=f'{prev_source} {chunk}'
        	ans =_translate_text_chunk_in_context(input_chunk,prev,tokenizer,model,num_beams=num_beams,max_new_tokens=max_new_tokens)
        	yield chunk,ans
        	prev=ans
        	prev_source=chunk

if __name__=="__main__":
	import sqlite3

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
	    LIMIT 10
	""")

	# Fetch the results
	english_texts = [x[0] for x in cursor.fetchall()]


	model,tokenizer=get_quantmodel_and_tokenizer()
	spliter=stanza.Pipeline(lang='en',verbose=False)

	for text in english_texts:
		for original,trans in gen_translate_text_pairs(text,spliter,tokenizer,model):
			print(original)
			print(2*"\n")
			print(trans)
			print(5*"\n")