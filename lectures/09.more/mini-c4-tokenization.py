import io
import pathlib
import sentencepiece
import tensorflow as tf
import keras
import keras_hub

TEXTGEN_DIR = pathlib.Path("text-generation")
TEXTGEN_DIR.mkdir(exist_ok=True)

extract_dir = keras.utils.get_file(
    fname="mini-c4",
    origin=("https://hf.co/datasets/mattdangerw/mini-c4/resolve/main/mini-c4.zip"),
    cache_dir=TEXTGEN_DIR,
    extract=True,
)
DS_DIR = pathlib.Path(extract_dir) / "mini-c4"

files = [str(file) for file in DS_DIR.glob("*.txt")]
ds = tf.data.Dataset.from_tensor_slices(files)


def read_file(filename):
    lines = tf.data.TextLineDataset(filename).filter(lambda x: tf.strings.length(x) > 0)
    # # avoid `Found null character.` warning
    # lines = lines.map(lambda x: tf.strings.regex_replace(x, r"\x00", ""))
    # '\n' are escaped in the dataset <-> one multiline document per line
    lines = lines.map(lambda x: tf.strings.regex_replace(x, r"\\n", "\n"))
    # each line is a 'document' -> append the 'end of document' token
    lines = lines.map(lambda x: tf.strings.join([x, "<|endoftext|>"]))
    return lines


ds = ds.interleave(read_file, cycle_length=32, num_parallel_calls=32)
bytes_io = io.BytesIO()
# options: https://github.com/google/sentencepiece/blob/master/doc/options.md
sentencepiece.SentencePieceTrainer.train(
    sentence_iterator=ds.as_numpy_iterator(),
    model_type="bpe",  # default is "unigram"
    model_writer=bytes_io,
    # size extracted using Matthew Watson's proto file -> tokenizer.vocabulary_size()
    vocab_size=32000,
    # we can add a special token ourselves (not in Watson's file)
    # user_defined_symbols=["<|endoftext|>"],
    # overkill: guaranteeing we catch even very long lines, default: 4192
    max_sentence_length=500_000,
    # no cap on # sentences used, adjust given available RAM (10_000_000 would be ok)
    input_sentence_size=0,
    # the default: random sample rather than just the first N files
    shuffle_input_sentence=True,
    # default: 0.9995
    character_coverage=0.99995,
    # make sure we have all 256 bytes in the vocab
    byte_fallback=True,
    # optional tweaks
    split_digits=True,
    allow_whitespace_only_pieces=True,
)
proto = bytes_io.getvalue()

# save proto
vocabulary_file = str(TEXTGEN_DIR / "vocabulary.from-scratch.proto")
with open(vocabulary_file, "wb") as o:
    o.write(proto)

tok = keras_hub.tokenizers.SentencePieceTokenizer(proto)

vocabulary_file_ref = keras.utils.get_file(
    origin="https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto",
)
tok_ref = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file_ref)

# get vocabs
ref_vocab = [tok_ref.id_to_token(i) for i in range(tok_ref.vocabulary_size())]
scratch_vocab = [tok.id_to_token(i) for i in range(tok.vocabulary_size())]

vocabs_path = TEXTGEN_DIR / "vocabs.txt"
print(f"saving vocabs to {vocabs_path}")
with open(vocabs_path, "w") as o:
    for i, (ref, scr) in enumerate(zip(ref_vocab, scratch_vocab)):
        o.write(f"{ref}\t{scr}\n")
