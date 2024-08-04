import torch
import torch.nn as nn
import streamlit as st


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, bias, context_length):
        super().__init__()
        self.d_out = d_out

        self.Wq = nn.Linear(d_in, d_out, bias)
        self.Wk = nn.Linear(d_in, d_out, bias)
        self.Wv = nn.Linear(d_in, d_out, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs, masking=True):
        batch_size, num_tokens, d_in = inputs.shape

        st.markdown("### Inputs to Self-Attention")
        st.write(f"Shape: `{inputs.shape}` (batch_size, num_tokens, d_in)")

        queries = self.Wq(inputs)
        st.markdown("### Queries")
        st.write(f"Shape: `{queries.shape}` (batch_size, num_tokens, d_in)")

        keys = self.Wk(inputs)
        st.markdown("### Keys")
        st.write(f"Shape: `{keys.shape}` (batch_size, num_tokens, d_in)")

        values = self.Wv(inputs)
        st.markdown("### Values")
        st.write(f"Shape: `{values.shape}` (batch_size, num_tokens, d_in)")

        st.markdown("### Attention Scores Calculation")
        st.code("attn_scores = queries @ keys.transpose(1, 2)", language="python")
        attn_scores = queries @ keys.transpose(1, 2)
        st.write(f"Shape: `{attn_scores.shape}` (batch_size, num_tokens, num_tokens)")

        with st.expander("View Attention Scores"):
            st.write(attn_scores)

        if masking:
            attn_scores.masked_fill_(
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
            )
            st.markdown("### Attention Scores with Masking")
            with st.expander("View Masked Attention Scores"):
                st.write(attn_scores)
            st.write(
                f"Shape after masking: `{attn_scores.shape}` (batch_size, num_tokens, num_tokens)"
            )

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        st.markdown("### Attention Weights")
        st.write(f"Shape: `{attn_weights.shape}` (batch_size, num_tokens, num_tokens)")

        context_vector = attn_weights @ values
        st.markdown("### Output from Self-Attention")
        st.write(f"Shape: `{context_vector.shape}` (batch_size, num_tokens, d_in)")

        return context_vector


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, bias=False, masking=True
    ):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.masking = masking

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.head_dim = d_out // num_heads
        self.Wq = nn.Linear(d_in, d_out, bias)
        self.Wk = nn.Linear(d_in, d_out, bias)
        self.Wv = nn.Linear(d_in, d_out, bias)
        self.out_proj = nn.Linear(d_out, d_out, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs):
        st.write("### Multi-Head Attention")
        st.write("    ", "#" * 10, "Start of Multi-Head Attention", "#" * 10)
        batch_size, num_tokens, d_in = inputs.shape
        st.write(f"(Inputs).shape: `{inputs.shape}` (batch_size, num_tokens, d_in)")

        keys = self.Wk(inputs)
        queries = self.Wq(inputs)
        values = self.Wv(inputs)
        st.write(f"---- (keys).shape:  `{keys.shape}` (batch, num_tokens, d_out)")
        st.write(f"---- (queries).shape:  `{queries.shape}` (batch, num_tokens, d_out)")
        st.write(f"---- (values).shape:  `{values.shape}` (batch, num_tokens, d_out)")
        d_k = keys.shape[-1]

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        st.info("After splitting heads:")
        st.write(
            f"---- (keys).shape:  `{keys.shape}` (batch_size, num_tokens, num_heads, head_dim)"
        )
        st.write(
            f"---- (queries).shape:  `{queries.shape}` (batch_size, num_tokens, num_heads, head_dim)"
        )
        st.write(
            f"---- (values).shape:  `{values.shape}` (batch_size, num_tokens, num_heads, head_dim)"
        )

        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        st.info("Bring num_heads before num_tokens:")
        st.write(
            f"---- (keys).shape:  `{keys.shape}` (batch_size, num_heads, num_tokens, head_dim)"
        )
        st.write(
            f"---- (queries).shape:  `{queries.shape}` (batch_size, num_heads, num_tokens, head_dim)"
        )
        st.write(
            f"---- (values).shape:  `{values.shape}` (batch_size, num_heads, num_tokens, head_dim)"
        )

        st.info("Calculate attention scores")
        st.code("attn_scores = queries @ (keys.transpose(2, 3))", language="python")
        attn_scores = queries @ (keys.transpose(2, 3))
        st.write(
            f"(attn_scores).shape:  `{attn_scores.shape}` (batch_size, num_heads, num_tokens, num_tokens)"
        )

        if self.masking:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            st.info("After masking:")
            st.write(
                f"(attn_scores).shape:  `{attn_scores.shape}` (batch_size, num_heada, num_tokens, num_tokens)"
            )

        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        st.write(
            f"(attn_weights).shape:  `{attn_weights.shape}` (batch_size, num_heada, num_tokens, num_tokens)"
        )
        attn_weights = self.dropout(attn_weights)
        st.write(
            f"(After dropout).shape:  `{attn_weights.shape}` (batch_size, num_heada, num_tokens, num_tokens)"
        )

        st.info("Calculate context vector = attn_weights @ values")
        context_vector = attn_weights @ values
        st.write(
            f"(context_vector).shape:  `{context_vector.shape} `(batch_size, num_heads, num_tokens, head_dim)"
        )

        st.info("Transpose back to (batch_size, num_tokens, num_heads, head_dim)")
        context_vector = context_vector.permute(0, 2, 1, 3)
        st.write(
            f"(context_vector).shape: `{context_vector.shape}` (batch_size, num_tokens, num_heads, head_dim)"
        )

        st.info("Flatten to (batch, num_tokens, d_out)")
        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.d_out
        )
        st.write(
            f"(After flattening).shape: `{context_vector.shape}` (batch_size, num_tokens, d_out)"
        )

        context_vector = self.out_proj(context_vector)
        st.write(
            f"(After final Linear).shape: `{context_vector.shape} `(batch_size, num_tokens, d_out)"
        )
        st.write(
            f"(Output).shape: `{context_vector.shape}` (batch_size, num_tokens, d_out)"
        )

        st.write("    ", "#" * 10, "End of Multi-Head Attention", "#" * 10)

        return context_vector


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["d_model"], 4 * cfg["d_model"]),
            GELU(),
            nn.Linear(4 * cfg["d_model"], cfg["d_model"]),
        )

    def forward(self, x):
        # st.write(f"Input.shape: ``{x.shape}`` (batch_size, num_tokens, d_in)")
        out = self.layers(x)
        # st.write(f"Output.shape: `{out.shape}` (batch_size, num_tokens, d_in)")
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_in=cfg["input_dim"],
            d_out=cfg["d_model"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            bias=cfg["bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["d_model"])
        self.norm2 = LayerNorm(cfg["d_model"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        st.write("### Transformer Block")
        st.write("  ", "<" * 10, "Transformer Block Start", ">" * 10)
        st.write(
            f"(Input to TransformerBlock).shape : `{x.shape}` (batch_size, num_tokens, d_in)"
        )

        shortcut = x
        st.write(
            f"(shortcut).shape : `{shortcut.shape}` (batch_size, num_tokens, d_in)"
        )

        x = self.norm1(x)
        st.write(
            f"(After LayerNorm1).shape : `{x.shape}` (batch_size, num_tokens, d_in)"
        )

        x = self.mha(x)
        st.write(
            f"(After MultiHead-Attention).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        x = self.dropout(x)
        st.write(
            f"(After Dropout).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        x = x + shortcut
        st.write(
            f"(After adding Residual).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        shortcut = x
        st.write(
            f"(shortcut).shape : `{shortcut.shape}`  (batch_size, num_tokens, d_out)"
        )

        x = self.norm2(x)
        st.write(
            f"(After LayerNorm2).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        x = self.ff(x)
        st.write(
            f"(After FeedForward).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        x = self.dropout(x)
        st.write(
            f"(After Dropout).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        x = x + shortcut
        st.write(
            f"(After adding Residual).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )

        st.write(
            f"(Output of TransformerBlock).shape : `{x.shape}`  (batch_size, num_tokens, d_out)"
        )
        st.write("  ", "<" * 10, "Transformer Block End", ">" * 10)

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["input_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["input_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["d_model"])
        self.out_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, tokenized_inputs):
        st.write("### GPT Model")
        st.write("+" * 10, "GPT START", "+" * 10)
        batch_size, seq_len = tokenized_inputs.shape
        st.write(
            f"Input to GPT.shape: `{tokenized_inputs.shape}` (batch_size, num_tokens)"
        )

        tok_embeds = self.tok_emb(tokenized_inputs)
        st.write(f"tok_embeds.shape: `{tok_embeds.shape}` (batch, num_tokens, d_in)")

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=tokenized_inputs.device))
        st.write(f"pos_embeds.shape: `{pos_embeds.shape}` (context_length , d_in)")

        x = tok_embeds + pos_embeds
        st.write(
            f"(tok_embeds + pos_embeds).shape: `{x.shape}` (batch, num_tokens, d_in)"
        )

        x = self.dropout(x)
        st.write(f"(After dropout).shape: `{x.shape}` (batch, num_tokens, d_in)")

        x = self.transformer_blocks(x)
        st.write(
            f"(After TransformerBlocks).shape: `{x.shape}` (batch, num_tokens, d_out)"
        )

        x = self.final_norm(x)
        st.write(
            f"(After final LayerNorm).shape: `{x.shape}` (batch, num_tokens, d_out)"
        )

        logits = self.out_head(x)
        st.write(
            f"(After linear).shape: `{logits.shape}` (batch, num_tokens, vocab_size)"
        )
        st.write(
            f"Output of GPT.shape: `{logits.shape}` (batch, num_tokens, vocab_size)"
        )
        st.write("+" * 10, "GPT END", "+" * 10)

        return logits
