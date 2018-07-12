""" Testing the functionality located in src/data/vocab """

import os

import pytest

from src.data.vocab import Vocab

WORKING_DIR = os.path.abspath(os.path.dirname(__file__)) # path to file
BASE_DIR = os.path.abspath(os.path.join(WORKING_DIR, "../../data"))
TEST_VOCAB_PATH = os.path.join(BASE_DIR, "processed/vocab")

@pytest.fixture
def empty_vocab():
    """ Returns a standard empty vocab """
    return Vocab()

@pytest.fixture
def loaded_vocab():
    """ Returns a vocab object that has loaded a vocab file """
    return Vocab(TEST_VOCAB_PATH)

def test_prep_train_seq(empty_vocab):
    sample_seq = "This is a sample sentence to test the method that builds the vocab one sentence at a time"
    # TODO: Add more samples here to test different types of sentences
    empty_vocab.prep_train_seq(sample_seq)
    sample_seq = sample_seq.lower().split(' ')
    for tok in sample_seq:
        assert tok in empty_vocab.vocab

def test_prep_seq(loaded_vocab):
    sample_seq = "This sentence will test included words in the dictionary"
    id_list = loaded_vocab.prep_seq(sample_seq)
    assert id_list[0] == 1 and id_list[-1] == 2

def test_unk_in_vocab(loaded_vocab):
    unk_seq = "This will test out of vocabulary words like phirr"
    id_list = loaded_vocab.prep_seq(unk_seq)
    assert 3 in id_list
