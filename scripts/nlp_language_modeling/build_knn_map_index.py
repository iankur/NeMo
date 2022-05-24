# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is the script to build KNN index map from Training dataset to Retrieval dataset.
For example, it maps chunk_id i from training dataset to K chunk ids in the nearest neighbor in the retireval dataset.

It requires the training text data to be converted into `bin` and `idx` files by `preprocess_data_for_megatron.py` script.
It also requires the Faiss Index file for the Retrieval dataset built by `build_retrieval_index.py` script.

Here is an example to using it:

```python
python scripts/nlp_language_modeling/build_knn_map_index.py \
    --input_file=PATH_TO_INPUT_TRAINING_DATA \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=tokenizer.model \
    --process_chunk_size=51200 \
    --K_neighbors=16 \
    --faiss_index=PATH_TO_FAISS_INDEX_FILE \
    --devices=0,1,2,3 \
    --batch_size=1280 \
    --output_file=knn_map.idx 
```

It creates a knn_map.idx KNNIndex file.
During training of RETRO model, it can look up the KNN chunk ids of the 
DB dataset given the input training data chunk id. 

"""
import argparse
import multiprocessing

import faiss
from sentence_transformers import SentenceTransformer

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    KNNIndex,
    MMapRetrievalIndexedDataset,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

queue = multiprocessing.Queue(10)
emb_queue = multiprocessing.Queue(10)


def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        vocab_file=args.vocab_file,
        merges_file=args.merge_file,
        delimiter=args.delimiter,
    )
    if not hasattr(tokenizer, "pad_id"):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    elif hasattr(tokenizer, "pad_id") and (tokenizer.pad_id is None or tokenizer.pad_id < 0):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer


def process_sentence_chunks(ds: MMapRetrievalIndexedDataset, tokenizer, chunk_size: int):
    total_chunks = ds.chunks
    start = 0
    threshold = 0
    while start < total_chunks:
        if start / total_chunks > threshold:
            logging.info(f"sentence processing {start / total_chunks} is done")
            threshold += 0.1
        id_slices = ds.get_chunk(slice(start, min(start + chunk_size, total_chunks)), force_no_padding=True)
        start = min(start + chunk_size, total_chunks)
        sentences = [tokenizer.ids_to_text(ids) for ids in id_slices]
        queue.put(sentences)
    queue.put(None)


def get_sentence_chunks():
    return queue.get()


def calculate_embedding(pool, batch_size):
    while True:
        sentences = get_sentence_chunks()
        if sentences is None:
            break
        emb = model.encode_multi_process(sentences=sentences, pool=pool, batch_size=batch_size)
        emb_queue.put(emb)
    emb_queue.put(None)


def get_emb():
    return emb_queue.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build Faiss index",)
    parser.add_argument(
        '--input_file', type=str, required=True, help='Input file',
    )
    parser.add_argument("--faiss_index", type=str, required=True, help='faiss index file for retrieval dataset')
    parser.add_argument(
        '--process_chunk_size',
        type=int,
        default=10000,
        help='The sentences in chunks that is queries to build map index',
    )
    parser.add_argument(
        '--K_neighbors', type=int, default=16, help='The number of neighbors to query',
    )
    parser.add_argument(
        '--sentence_transformer_model',
        type=str,
        default='bert-base-nli-mean-tokens',
        help='sentence transformer to load',
    )
    parser.add_argument(
        '--output_file', type=str, required=True, help='Output KNN Map index file',
    )
    parser.add_argument(
        '--devices', type=str, default=None, help='delimited list input with cuda devices. Specify like 0,1,2'
    )
    parser.add_argument(
        "--batch_size", type=int, default=4000, help="Batch size for encoding. Use max according to GPU MEM"
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-library',
        type=str,
        required=True,
        choices=['yttm', 'sentencepiece', 'megatron', 'huggingface', 'tabular'],
        help='What tokenizer library to use.',
    )
    group.add_argument(
        '--tokenizer-type', type=str, default=None, help='What type of tokenizer to use.',
    )
    group.add_argument(
        '--tokenizer-model', type=str, default=None, help='Path to tokenizer model.',
    )
    group.add_argument('--vocab-file', type=str, default=None, help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None, help='Path to the BPE merge file (if necessary).')
    group.add_argument('--delimiter', type=str, default=None, help='delimiter used for tabular tokenizer')

    args = parser.parse_args()
    model = SentenceTransformer(args.sentence_transformer_model)
    tokenizer = get_tokenizer(args)
    ds = MMapRetrievalIndexedDataset(args.input_file)
    index = faiss.read_index(args.faiss_index)

    # make sure the dataset is padded as retrieval database
    assert not ds._index.retrieval_db

    process = multiprocessing.Process(target=process_sentence_chunks, args=(ds, tokenizer, args.process_chunk_size))
    process.start()

    if args.devices is None:
        device_list = None
    else:
        device_list = ['cuda:' + str(device) for device in args.devices.split(',')]

    pool = model.start_multi_process_pool(device_list)

    emb_process = multiprocessing.Process(target=calculate_embedding, args=(pool, args.batch_size))
    emb_process.start()

    with KNNIndex.writer(args.output_file, args.K_neighbors) as w:
        while True:
            emb = get_emb()
            if emb is None:
                break
            D, I = index.search(emb, args.K_neighbors)
            w.write(I)

    process.join()
    emb_process.join()
    model.stop_multi_process_pool(pool)