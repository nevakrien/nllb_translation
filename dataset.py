from datasets import concatenate_datasets, load_dataset

def load_data():
    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

    assert bookcorpus.features.type == wiki.features.type
    raw_datasets = concatenate_datasets([bookcorpus, wiki])
    
    return raw_datasets


if __name__=="__main__":
    #print(dir(load_data))
    data=load_data()

    #print(data[1])
    #    for i in range(10):
    #        print(data[i])
    #     
    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

    print('wiki data')
    print(wiki[0])


    print('book data')
    print(bookcorpus[0])

