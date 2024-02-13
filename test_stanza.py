import stanza
# Initialize the English NLP pipeline
nlp = stanza.Pipeline(lang='en',verbose=False)
#print(nlp)
#print(help(stanza.Pipeline.__init__))
# Process some text
doc = nlp("Stanford University is located in California. It's known for its academic strength.")

# Iterate over sentences and print them
for sentence in doc.sentences:
    print(sentence.text)

# # Access tokens and their part-of-speech tags
# for sentence in doc.sentences:
#     for token in sentence.tokens:
#         print(f'{token.text}/{token.words[0].pos}')