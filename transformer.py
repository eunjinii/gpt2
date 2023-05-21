import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe


from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import tempfile

from functions import Encoder, Decoder


train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(
    map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda")


def batchify(data, bsz):
    # Divide the dataset into ``bsz`` parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the ``bsz` batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 25


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data.t(), target


ntokens = len(vocab)  # the size of vocabulary
emsize = 4096  # embedding dimension
nhid = 4096  # the dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 12  # the number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 16  # the number of heads in the Multihead Attention models
dropout = 0.2  # the dropout value

tmpfile = tempfile.NamedTemporaryFile()
rpc.init_rpc(
    name="worker",
    rank=0,
    world_size=1,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
        # Specifying _transports and _channels is a workaround and we no longer
        # will have to specify _transports and _channels for PyTorch
        # versions >= 1.8.1
        _transports=["ibv", "uv"],
        _channels=["cuda_ipc", "cuda_basic"],
    )
)

num_gpus = 2
partition_len = ((nlayers - 1) // num_gpus) + 1

# Add encoder in the beginning.
tmp_list = [Encoder(ntokens, emsize, dropout).cuda(0)]
module_list = []

# Add all the necessary transformer blocks.
for i in range(nlayers):
    transformer_block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    if i != 0 and i % (partition_len) == 0:
        module_list.append(nn.Sequential(*tmp_list))
        tmp_list = []
    device = i // (partition_len)
    tmp_list.append(transformer_block.to(device))

# Add decoder in the end.
tmp_list.append(Decoder(ntokens, emsize).cuda(num_gpus - 1))
module_list.append(nn.Sequential(*tmp_list))


# Build the pipeline.
chunks = 8
model = Pipe(torch.nn.Sequential(*module_list), chunks=chunks)


def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


print('Total parameters in model: {:,}'.format(get_total_params(model)))
