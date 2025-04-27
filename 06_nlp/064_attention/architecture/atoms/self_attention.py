import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size):
        # embed_dim son las dimensiones del embedding de los tokens originalmente
        # head_dim son las dimensiones del vector resultante del mecanismo de atención
        # block_size son el número de tokens que hacen parte del contexto
        super().__init__()
        self.head_dim = head_dim

        self.Wq = nn.Linear(embed_dim, head_dim, bias=False) # Matriz para generar la representación Q de los tokens
        self.Wk = nn.Linear(embed_dim, head_dim, bias=False) # Matriz para generar la representación K de los tokens
        self.Wv = nn.Linear(embed_dim, head_dim, bias=False) # Matriz para generar la representación V de los tokens

        # Indicar a pytorch que la máscara no hace parte de los parámetros del modelo
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size))) # Máscara para asegurarse de que los tokens no presten atención a tokens futuros

    def forward(self, x):  # x.shape = [batch_size, block_size, embed_dim]
        N, T, D = x.shape
        # Crear representaciones de los tokens
        Q = self.Wq(x)  # [N, T, D] @ [D, head_dim] = [N, T, head_dim]
        K = self.Wk(x)  # [N, T, D] @ [D, head_dim] = [N, T, head_dim]
        V = self.Wv(x)  # [N, T, D] @ [D, head_dim] = [N, T, head_dim]
        # Calcular puntajes de similitud
        att_weights = Q @ K.transpose(-1, -2)  # [N, T, head_dim] @ [N, head_dim, T] = [N, T, T]
        att_weights = att_weights * self.head_dim ** -0.5  # Reducir el tamaño de los puntajes de similitud
        # Enmascarar los tokens futuros. El puntaje de similitud de los tokens futuros se transforma a -inf de modo que la función Softmax los convierta a 0
        masked_att = att_weights.masked_fill(self.mask[:T, :T] == 0, -torch.inf)
        att_weights = torch.nn.functional.softmax(masked_att, dim=2)
        self.att_weights = att_weights  # Esto se agrega para visualizar los resultados de la atención más adelante
        # Resultado del mecanismo de atención
        weighted_output = att_weights @ V  # [N, T, T] @ [N, T, head_dim] = [N, T, head_dim]

        return weighted_output