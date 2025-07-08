"""
Codes taken and modified from https://github.com/wjq-learning/CBraMod

A EEG Foundation model for classification and regression.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional, Union, Callable


class CBraMod(nn.Module):
    """
    CBraMod model for EEG signal processing using a transformer-based architecture.

    Input shape: (batch_size, num_of_channels, time_segments, in_dim)
    Patch Embedding: (batch_size, num_of_channels, time_segments, d_model)
    Transformer Encoded Embedding: (batch_size, num_of_channels, time_segments, d_model)
    Output shape: (batch_size, num_of_channels, time_segments, out_dim)

    Attributes
    ----------
    patch_embedding : PatchEmbedding
        Module for embedding EEG patches into a higher-dimensional space.
    encoder : TransformerEncoder
        Transformer encoder for processing the embedded patches.
    proj_out : nn.Sequential
        Module for projecting the output of the transformer encoder
        to the desired output dimension.
    
    Methods
    -------
    forward(x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor
        Forward pass of the model.
        Takes an input tensor and applies patch embedding,
        transformer encoding, and projection to output.
    """
    def __init__(
            self, in_dim: int=200, out_dim: int=200,
            d_model: int=200, dim_feedforward: int=800,
            seq_len: int=30, n_layer: int=12,
            nhead: int=8, linear_proj: bool=True
        ):
        """
        Parameters
        ----------
        in_dim : int
            Input dimension. (last dimension of the tensor)
            This represents number of samples in a time window.
        out_dim : int
            Output dimension of the model.
            The last dimension of the output tensor.
        d_model : int
            Dimension of the latent space of the transformer encoder.
        dim_feedforward : int
            Dimension of the feedforward network in the transformer encoder.
        seq_len : int
            Length of the input sequence.
        n_layer : int
            Number of layers in the transformer encoder.
        nhead : int
            Number of attention heads in the transformer encoder.
        linear_proj : bool, optional
            If True, use a simple linear projection for the output.
            If False, use a more complex projection with multiple layers
            and GELU activation. \n
            Default is True.
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation=F.gelu 
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers=n_layer)
    
        if linear_proj:
            self.proj_out = nn.Sequential(
                nn.Linear(d_model, out_dim),
            )
        else:
            self.proj_out = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                nn.GELU(),
                nn.Linear(d_model*2, d_model),
                nn.GELU(),
                nn.Linear(d_model, out_dim),
            )
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape 
            (batch_size, num_of_channels, time_segments, in_dim).
            This corresponds to the EEG signal.
        mask : torch.Tensor, optional
            Mask tensor of the same shape as x,
            where 1 indicates masked positions.
            If None, no masking is applied.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape 
            (batch_size, num_of_channels, time_segments, out_dim).
            This corresponds to the embedded features. 
        """
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out


class PatchEmbedding(nn.Module):
    """
    Embed EEG patches into a higher-dimensional space
    using convolutional layers and spectral features.
    """
    def __init__(self, in_dim: int, out_dim: int, d_model: int, seq_len: int):
        """
        Parameters
        ----------
        in_dim : int
            Input dimension. (last dimension of the tensor)
        out_dim : int
            Output dimension of the model. (placeholder)
        d_model : int
            Dimension of the output features.
            This means the dimension of the output embeddings.
        seq_len : int
            Length of the input sequence. (placeholder)
        """
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model, out_channels=d_model,
                kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                groups=d_model
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        # Projection layers to embed patches
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(
            self, x: torch.Tensor,
            mask: torch.Tensor=None
        ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape 
            (batch_size, num_of_channels, time_segments, points_per_patch).
            This corresponds to the EEG signal.
        mask : torch.Tensor, optional
            Mask tensor of the same shape as x,
            where 1 indicates masked positions.
            If None, no masking is applied.

        Returns
        -------
        torch.Tensor
            Output tensor of shape 
            (batch_size, num_of_channels, time_segments, d_model).
            This corresponds to the embedded features.
        """
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    """
    Initialise weights for the model layers
    using Kaiming normal initialisation for linear and convolutional layers,
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    This layer consists of two multi-head self-attention mechanisms
    and a feedforward neural network, with layer normalisation and dropout.
    Self-attention is applied on temporal and spatial dimensions separately.

    Attributes
    -----------
    self_attn_s : nn.MultiheadAttention
        Multi-head self-attention mechanism for spatial attention.
    self_attn_t : nn.MultiheadAttention
        Multi-head self-attention mechanism for temporal attention.
    linear1 : nn.Linear
        Linear layer for the self-attention block
    linear2 : nn.Linear
        Linear layer for the feedforward (fully-connected) block.
    norm1 : nn.LayerNorm
        Layer normalisation applied before self-attention block
    norm2 : nn.LayerNorm
        Layer normalisation applied before feedforward block
    dropout1 : nn.Dropout
        Dropout layer applied after self-attention block
    dropout2 : nn.Dropout
        Dropout layer applied after feedforward block
    activation : Callable[[Tensor], Tensor]
        Activation function used in the feedforward network.
    activation_relu_or_gelu : int
        An integer indicating the type of activation function used:
        1 for ReLU, 2 for GELU, and 0 for other functions.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False,
                bias: bool = True, device: Optional[torch.device]=None,
                dtype: Optional[torch.dtype]=None) -> None:
        """
        Parameters
        ----------
        d_model : int
            Dimension of the input and output features.
        nhead : int
            Total number of attention heads.
        dim_feedforward : int, optional
            Dimension of the hidden layero of feedforward network
            (default is 2048).
        dropout : float, optional
            Dropout probability (default is 0.1).
        activation : Union[str, Callable[[Tensor], Tensor]], optional
            Activation function to use in the feedforward network.
            It can be a string ('relu', 'gelu') or a callable function.
            Default is F.relu.
        layer_norm_eps : float, optional
            Value added on the standard deviation in layer normalisation
            to avoid division by zero.
            (default is 1e-5).
        batch_first : bool, optional
            Used for Multi-headed attention.
            If True, the input and output tensors are of shape
            (batch_size, seq_len, d_model).
            Otherwise, the shapes are
            (seq_len, batch_size, d_model).
        bias : bool, optional
            Whether to add a bias term in the linear layers.
            Default is True.
        device : Optional[torch.device], optional
            The device on which the module will be allocated.
            If None, the default device is used.
        dtype : Optional[torch.dtype], optional
            The data type of the module's parameters.
            If None, the default data type is used.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # spatial self-attention
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        # temporal self-attention
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        src : Tensor
            Input tensor of shape (batch_size, num_of_channels, patch_num, d_model).
            This corresponds to the EEG signal.
        src_mask : Optional[Tensor]
            Attention mask tensor of shape (patch_num, patch_num)
            or (batch_size, patch_num, patch_num).
            If None, no attention mask is applied.
        src_key_padding_mask : Optional[Tensor]
            Key padding mask tensor of shape (batch_size, patch_num).
            (if masked, that token is ignored in the attention).
            If None, no key padding mask is applied.
        """

        x = src
        # residual self-attention
        x = x + self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask)
        # residual feedforward (Fully-connected) network
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(
            self, x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, num_of_channels, patch_num, d_model).
        attn_mask : Optional[Tensor]
            Attention mask tensor of shape (patch_num, patch_num)
            or (batch_size, patch_num, patch_num).
            If None, no attention mask is applied.
        key_padding_mask : Optional[Tensor]
            Key padding mask tensor of shape (batch_size, patch_num).
            If None, no key padding mask is applied.
        
        Return
        -------
        Tensor
            Output tensor of the same shape as x.
        """
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(
            bz*patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(
            bz*ch_num, patch_num, patch_size // 2)
        # attention applied across channels
        xs = self.self_attn_s(xs, xs, xs,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)
        # attention applied across patches (time segments)
        xt = self.self_attn_t(xt, xt, xt,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape
            (batch_size, num_of_channels, patch_num, d_model).
        
        Return
        -------
        Tensor
            Output tensor of shape (batch_size, num_of_channels, patch_num, d_model).
            the feedforward (fully-connected) network.
        """
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    """
    A class used for assembling stack of TransformerEncoderLayer.
    """
    def __init__(
            self, encoder_layer: TransformerEncoderLayer,
            num_layers: int,
            norm: Optional[nn.Module]=None) -> None:
        """
        Parameters
        ----------
        encoder_layer : nn.Module
            An instance of TransformerEncoderLayer to be cloned.
        num_layers : int
            The number of sub-encoder-layers in the encoder.
        norm : Optional[nn.Module]
            An optional module for final normalisation of the output.
        """
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        src : Tensor
            Input tensor of shape (batch_size, num_of_channels, patch_num, d_model).
        mask : Optional[Tensor]
            Attention mask tensor of shape (patch_num, patch_num)
            or (batch_size, patch_num, patch_num).
            If None, no attention mask is applied.
        src_key_padding_mask : Optional[Tensor]
            Currently a place-holder
            Key padding mask tensor of shape (batch_size, patch_num).
            If None, no key padding mask is applied.

        Returns
        -------
        Tensor
            Output tensor of the same shape as src.
        """

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(
            f"activation should be relu/gelu, not {activation}")
