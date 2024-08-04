import torch
import torch.nn as nn
import streamlit as st
import tiktoken


from GPTmodel import (
    GPTModel,
    TransformerBlock,
    MultiHeadAttention,
    LayerNorm,
    GELU,
    FeedForward,
    SelfAttention,
)

"""        
GPT_CONFIG = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "input_dim": embed_dim,
            "d_model": embed_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "drop_rate": drop_rate,
            "bias": bias,
        }
"""


def input_text_to_input_embeddings(input_text: str, GPT_CONFIG) -> torch.Tensor:

    st.header("Input Text")
    st.write(input_text)
    st.markdown("---")

    st.header("Tokenize")
    with st.echo():
        tokenized = input_text.split()

    st.write(tokenized)
    st.markdown("---")

    st.header("Tokenized IDs")

    st.subheader("Encoded Text")
    with st.echo():
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded_text = tokenizer.encode(input_text)
    st.write(encoded_text)

    st.subheader("Encoded Tensor")
    with st.echo():
        encoded_tensor = torch.tensor(encoded_text).unsqueeze(dim=0)
    st.write(encoded_tensor)
    st.info(f"Shape: {encoded_tensor.shape}")
    st.markdown("---")

    st.header("Token Embeddings")

    st.subheader("1. Token Embeddings")
    with st.echo():
        tok_emb_layer = nn.Embedding(
            num_embeddings=GPT_CONFIG["vocab_size"],
            embedding_dim=GPT_CONFIG["input_dim"],
        )
    st.write(tok_emb_layer)

    st.subheader("Encoded Tensor passed to Token Embedding Layer")
    with st.echo():
        tok_embeds = tok_emb_layer(encoded_tensor)
    st.write(tok_embeds)
    st.info(f"Shape: {tok_embeds.shape}")
    st.markdown("---")

    st.header("Positional Embeddings")

    st.subheader("2. Positional Embeddings")
    with st.echo():
        pos_emb_layer = nn.Embedding(
            num_embeddings=GPT_CONFIG["context_length"],
            embedding_dim=GPT_CONFIG["input_dim"],
        )
    st.write(pos_emb_layer)

    with st.echo():
        seq_len = encoded_tensor.shape[-1]
        pos_embeds = pos_emb_layer(torch.arange(seq_len))
    st.info(f"Shape: {pos_embeds.shape}")
    st.markdown("---")

    st.header("Input Embeddings")

    st.subheader("3. Input Embeddings")
    with st.echo():
        input_embeds = tok_embeds + pos_embeds
    st.write(input_embeds)
    st.info(f"Shape: {input_embeds.shape}")
    st.markdown("---")

    st.success(
        f"We have now converted the text:\n\n"
        f"'{input_text}'\n\n"
        f"with token embedding layer and positional embedding layer\n\n"
        f"to input tensor with shape:\n\n{input_embeds.shape}"
    )

    return input_embeds, encoded_tensor


def understand_self_attention(input_embeddings, GPT_CONFIG):

    with st.echo():
        d_in = GPT_CONFIG["input_dim"]
        d_out = GPT_CONFIG["d_model"]
        dropout = GPT_CONFIG["drop_rate"]
        context_length = GPT_CONFIG["context_length"]
        bias = GPT_CONFIG["bias"]

        input_embeddings.shape
        sa = SelfAttention(d_in, d_out, dropout, bias, context_length)
        output = sa(input_embeddings, masking=False)
        output.shape


def understand_masked_self_attention(input_embeddings, GPT_CONFIG):

    with st.echo():
        d_in = GPT_CONFIG["input_dim"]
        d_out = GPT_CONFIG["d_model"]
        dropout = GPT_CONFIG["drop_rate"]
        context_length = GPT_CONFIG["context_length"]
        bias = GPT_CONFIG["bias"]

        input_embeddings.shape
        sa = SelfAttention(d_in, d_out, dropout, bias, context_length)
        output = sa(input_embeddings, masking=True)
        output.shape


def understand_multihead_masked_self_attention(input_embeddings, GPT_CONFIG):

    with st.echo():
        d_in = GPT_CONFIG["input_dim"]
        d_out = GPT_CONFIG["d_model"]
        dropout = GPT_CONFIG["drop_rate"]
        context_length = GPT_CONFIG["context_length"]
        bias = GPT_CONFIG["bias"]
        num_heads = GPT_CONFIG["n_heads"]

        input_embeddings.shape
        mha = MultiHeadAttention(
            d_in, d_out, context_length, dropout, num_heads, masking=True
        )

        output = mha(input_embeddings)
        output.shape


def understand_feedforward(input_embeddings, GPT_CONFIG):

    with st.echo():
        st.info(input_embeddings.shape)
        feedforward = FeedForward(GPT_CONFIG)
        output = feedforward(input_embeddings)
        st.info(output.shape)


def understand_layernorm(input_embeddings, GPT_CONFIG):

    with st.echo():
        st.info(input_embeddings.shape)
        layernorm = LayerNorm(d_model=GPT_CONFIG["d_model"])
        output = layernorm(input_embeddings)
        st.info(output.shape)


def understand_transformer(input_embeddings, GPT_CONFIG):

    with st.echo():
        st.info(input_embeddings.shape)
        trf_blk1 = TransformerBlock(GPT_CONFIG)
        output = trf_blk1(input_embeddings)
        st.info(output.shape)


def understand_GPT2(encoded_tensor, GPT_CONFIG):

    with st.echo():
        st.success(encoded_tensor)
        st.info(encoded_tensor.shape)
        gpt_model = GPTModel(GPT_CONFIG)
        output = gpt_model(encoded_tensor)
        st.info(output.shape)

    return output
