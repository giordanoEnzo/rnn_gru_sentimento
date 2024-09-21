import random
import torch
from jinja2 import optimizer
from torch import nn, optim
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from processamento.preparar_texttorch import tokenize, construir_vocabulario, tokens_to_ids, pad_sequences, \
    preparar_dataloader, dividir_treino_validacao

# %%
df_training = pd.read_csv('twitter_training.csv')
df_training = df_training.dropna(subset=[df_training.columns[2], df_training.columns[3]])
dados_treinamento = [
    (row[df_training.columns[2]], tokenize(row[df_training.columns[3]]))
    for _, row in df_training.iterrows()
    if isinstance(row[df_training.columns[3]], str)
]

df_test = pd.read_csv('twitter_validation.csv')
df_test = df_test.dropna(subset=[df_test.columns[2], df_test.columns[3]])
dados_teste = [
    (row[df_test.columns[2]], tokenize(row[df_test.columns[3]]))
    for _, row in df_test.iterrows()
    if isinstance(row[df_test.columns[3]], str)
]
# %%
print(dados_treinamento[0])
# %%
vocabulario = construir_vocabulario(dados_treinamento)
print(f"Tamanho do vocabulário: {len(vocabulario)}")
print(f"Exemplo de vocabulário: {list(vocabulario.items())[:10]}")
# %%
dados_treinamento_ids = tokens_to_ids(dados_treinamento, vocabulario)
dados_teste_ids = tokens_to_ids(dados_teste, vocabulario)

print(dados_treinamento_ids[0])
# %%
max_len = 64
dados_treinamento_padded = pad_sequences(dados_treinamento_ids, max_len)
dados_teste_padded = pad_sequences(dados_teste_ids, max_len)

print(dados_treinamento_padded[0])
# %%
dados_treinamento_final, dados_validacao = dividir_treino_validacao(dados_teste_padded)

print(f"Treino: {len(dados_treinamento_final)}, Validação: {len(dados_validacao)}")
# %%
batch_size = 64
train_loader = preparar_dataloader(dados_treinamento_final, batch_size, drop_last=True)
valid_loader = preparar_dataloader(dados_validacao, batch_size, drop_last=True)
test_loader = preparar_dataloader(dados_teste_padded, batch_size)

for batch in train_loader:
    print(batch)
    break


# %%
class RNN(nn.Module):

    def __init__(self, tam_vocab, tam_embedding, embed_vectors,
                 ind_unk, ind_pad, hidden_size, output_size):
        super(RNN, self).__init__()

        # Inicializaremos a camada de embedding
        self.embedding = nn.Embedding(tam_vocab, tam_embedding)
        self.embedding.weight.data.copy_(embed_vectors)
        self.embedding.weight.data[ind_unk] = torch.zeros(tam_embedding)
        self.embedding.weight.data[ind_pad] = torch.zeros(tam_embedding)
        #######################################

        self.hidden_size = hidden_size
        self.rnn = nn.GRU(tam_embedding, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, X, tamanhos):
        embed = self.embedding(X)

        # Inicializar a memória
        hidden = torch.zeros(1, X.size(1), self.hidden_size)

        # Empacotar a sequencia antes de alimentar a uniadde recorrente
        packed_input = nn.utils.rnn.pack_padded_sequence(embed, tamanhos)

        # Forward recorrente
        packed_output, hidden = self.rnn(packed_input, hidden)

        # Desempacotar a sequência para continuar o fluxo na rede
        output, output_lenghts = nn.utils.rnn.pad_packed_sequence(packed_output)

        output = self.linear(hidden.squeeze())

        return output


tam_vocab = len(vocabulario)
tam_embedding = 100  # tamanho do embedding
embed_vectors = torch.rand(tam_vocab, tam_embedding)  # inicialização
ind_unk = vocabulario.get('<unk>', 0)  # Índice para <unk>
ind_pad = vocabulario.get('<pad>', 0)  # Índice para <pad>
hidden_size = 256  # tamanho do estado oculto (Neurônios ocultos)
output_size = 3  # Exemplo de número de classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(tam_vocab, tam_embedding, embed_vectors, ind_unk, ind_pad, hidden_size, output_size)
model.to(device)
print(model)
# %%
criterio = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)


# %%
def forward(iterator, num_amostra, etapa):
    if etapa == 'treino':
        model.train()
    else:
        model.eval()

    acuracia = 0.
    loss_epoca = []
    for k, amostra in enumerate(iterator):
        texto, rotulo = amostra  # Descompactar diretamente
        tamanhos = [len(t) for t in texto]  # Exemplo para obter tamanhos das sequências

        saida = model(texto, tamanhos)

        loss = criterio(saida, rotulo)
        loss_epoca.append(loss.detach().cpu().numpy())

        _, pred = torch.max(saida, axis=-1)
        acuracia += (pred.cpu().data == rotulo.cpu().data).sum()

        if etapa == 'treino':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_epoca = np.asarray(loss_epoca).ravel()
    acuracia = acuracia / float(num_amostra)
    print('\n', '*' * 15 + etapa + '*' * 15)
    print('Epoca: {:}, Loss: {:.4f} +/- {:.4f}, Acurácia: {:.4f}'.format(
        epoca, loss_epoca.mean(), loss_epoca.std(), acuracia
    ))

    return loss_epoca.mean(), acuracia


# %%
loss_treino, loss_test = [], []
acc_treino, acc_test = [], []

for epoca in range(25):
    loss, acuracia = forward(train_loader, len(dados_treinamento_final), 'treino')
    loss_treino.append(loss)
    acc_treino.append(acuracia)

    loss, acuracia = forward(valid_loader, len(dados_validacao), 'teste')
    loss_test.append(loss)
    acc_test.append(acuracia)
# %%
nlp = spacy.load('en_core_web_sm')  # Certifique-se de ter esse modelo instalado


def predict_sentiment(sentence):
    model.eval()

    # Tokenizar a frase usando spaCy
    tokenized = [str(tok) for tok in nlp(sentence)]
    print("Tokenized:", tokenized)

    # Converter tokens em IDs usando o vocabulário
    indexed = [vocabulario.get(t, vocabulario['<unk>']) for t in tokenized]  # Usar <unk> para palavras desconhecidas
    if len(indexed) == 0:
        raise ValueError("A frase não contém palavras do vocabulário.")

    # Comprimento da sequência
    length = [len(indexed)]
    print("Indexed:", indexed)

    # Criar tensor e adicionar dimensão para batch
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)  # Adicionando dimensão para batch_size = 1
    length_tensor = torch.LongTensor(length).to(device)

    # Fazer a predição
    with torch.no_grad():
        prediction = model(tensor, length_tensor)

    return F.softmax(prediction, dim=-1).cpu().numpy()  # Retornar como numpy


# Testar a função com frases de exemplo
exemplos = [
    "I love this product!",
    "This is the worst experience I've ever had.",
    "It's okay, not great but not bad either.",
    "Absolutely fantastic service!",
    "I'm very disappointed with the quality."
]

# Gerar previsões e plotar os resultados
for frase in exemplos:
    pred = predict_sentiment(frase)

    # Plotar os resultados
    plt.bar(0, pred[0], color='darkred', label='Negativo', width=0.5)
    plt.bar(1, pred[1], color='dodgerblue', label='Positivo', width=0.5)
    plt.title(f"Sentiment Prediction for: '{frase}'")
    plt.legend()
    plt.show()