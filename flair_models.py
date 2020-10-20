from flair.embeddings import CharacterEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings,BytePairEmbeddings,DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam
from flair.datasets import CSVClassificationCorpus


for lang in ['malayalam','tamil']:
    corpus = CSVClassificationCorpus('data', train_file=f'train_{lang}.tsv', test_file=f'{lang}_dev.tsv',
                                 dev_file=f'dev_{lang}.tsv',delimiter='\t', skip_header=True,
                                 column_name_map={0:'text',1:'label_topic'})
    label_dict = corpus.make_label_dictionary()
    word_embeddings = [CharacterEmbeddings()]
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
    #trainer = ModelTrainer(classifier, corpus)
    trainer = ModelTrainer(classifier, corpus, optimizer=Adam)
    trainer.train(f'models/char/{lang}',learning_rate=0.1,mini_batch_size=32,anneal_factor=0.5,patience=5,max_epochs=20)
