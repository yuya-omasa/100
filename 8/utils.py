from gensim.models import KeyedVectors
import torch

def load_pretrained_embeddings(path, vocab_size, embedding_dim):
    # 事前学習済みword2vecモデルをロード（バイナリ形式に注意）
    wv = KeyedVectors.load_word2vec_format(path, binary=True)

    # 埋め込み行列の初期化（先頭行は <PAD> 用のゼロベクトル）
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))

    # 単語 ↔ ID 対応辞書の初期化
    word2id = {"<PAD>": 0}
    id2word = {0: "<PAD>"}

    for i, word in enumerate(wv.index_to_key[:vocab_size - 1]):
        idx = i + 1  # 0は<PAD>用に予約
        embedding_matrix[idx] = torch.tensor(wv[word])
        word2id[word] = idx
        id2word[idx] = word

    return embedding_matrix, word2id, id2word
