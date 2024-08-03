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

from understand_utils import (
    input_text_to_input_embeddings,
    understand_self_attention,
    understand_masked_self_attention,
    understand_multihead_masked_self_attention,
    understand_layernorm,
    understand_feedforward,
    understand_GPT2,
    understand_transformer,
)


def main():
    st.set_page_config(page_title="GPT-2 Demo", layout="wide")

    st.title("GPT-2 Model Demonstration")
    st.markdown("### A simple demo to understand the inner workings of GPT-2")

    st.markdown(
        """
    **Figure credit:**
    [ResearchGate: FLUID-GPT](https://www.researchgate.net/publication/373352176_FLUID-GPT_Fast_Learning_to_Understand_and_Investigate_Dynamics_with_a_Generative_Pre-Trained_Transformer_Efficient_Predictions_of_Particle_Trajectories_and_Erosion)
    """
    )
    st.image("GPT2_model.png")

    # Sidebar for input text and GPT configuration
    with st.sidebar:
        st.title("Configuration Panel")
        st.header("Settings")
        input_text = st.text_input(
            "Enter a sentence (e.g., I have a pen)", key="input_text"
        )

        st.markdown("### GPT Configuration")
        vocab_size = 50257  # Fixed value, explanation not needed

        context_length = st.number_input(
            "Context Length",
            min_value=1,
            max_value=2048,
            value=1024,
            step=1,
            help="The maximum length of the input sequence. Longer sequences will be truncated.",
        )
        embed_dim = st.number_input(
            "Embedding Dimension",
            min_value=6,
            max_value=2048,
            value=256,
            step=1,
            help="The size of the vector space in which words are embedded. This value must be divisible by the number of attention heads.",
        )
        n_heads = st.number_input(
            "Number of Heads",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
            help="The number of attention heads in each attention layer. The embedding dimension must be divisible by this value.",
        )

        if embed_dim % n_heads != 0:
            st.error("Embedding Dimension must be divisible by Number of Heads.")

        n_layers = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=24,
            value=6,
            step=1,
            help="The number of transformer blocks in the model.",
        )

        drop_rate = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="The dropout rate to prevent overfitting.",
        )
        bias = st.checkbox(
            "Use Bias",
            value=False,
            help="Whether to use bias terms in the linear transformations.",
        )

        GPT_CONFIG = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embed_dim": embed_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "drop_rate": drop_rate,
            "bias": bias,
        }

        st.markdown("### Analysis Steps")
        analysis_step2_cb = st.checkbox("Transformer Block Analysis", value=True)
        analysis_step3_cb = st.checkbox("Generate GPT Output", value=True)

    st.markdown("## Tokenization and Embedding")
    st.markdown(
        "Enter your text in the sidebar to start the tokenization and embedding process."
    )

    if input_text:
        with st.spinner("Processing..."):
            input_embeddings, encoded_tensor = input_text_to_input_embeddings(
                input_text, GPT_CONFIG
            )

        st.markdown("### Input Text")
        st.write(input_text)

        st.markdown("### Tokenized IDs")
        st.write(encoded_tensor)

        st.markdown("### Token Embeddings")
        st.write(input_embeddings.shape)

        # Always display Step 2 title and message, and optionally the tabs
        st.markdown("---")
        st.markdown("## Transformer Block Analysis")
        st.markdown(
            "Analyze different components of the transformer block based on your selection in the sidebar."
        )

        if analysis_step2_cb:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "Self Attention",
                    "Masked Self Attention",
                    "Multi-Head Masked Self Attention",
                    "FeedForward",
                    "LayerNorm",
                    "Transformer Block",
                ]
            )

            with tab1:
                st.markdown("### 2.1 Understand Self Attention")
                understand_self_attention(input_embeddings, GPT_CONFIG)

            with tab2:
                st.markdown("### 2.2 Understand Masked Self Attention")
                understand_masked_self_attention(input_embeddings, GPT_CONFIG)

            with tab3:
                st.markdown("### 2.3 Understand Multi-Head Masked Self Attention")
                understand_multihead_masked_self_attention(input_embeddings, GPT_CONFIG)

            with tab4:
                st.markdown("### 2.4 FeedForward")
                understand_feedforward(input_embeddings, GPT_CONFIG)

            with tab5:
                st.markdown("### 2.5 LayerNorm")
                understand_layernorm(input_embeddings, GPT_CONFIG)

            with tab6:
                st.markdown("### 2.6 Transformer Block")
                understand_transformer(input_embeddings, GPT_CONFIG)

        # Always display Step 3 title and message, and optionally the output
        st.markdown("---")
        st.markdown("## GPT Model Output")
        st.markdown("Generate and analyze the output of the GPT model.")

        if analysis_step3_cb:
            with st.spinner("Generating output..."):
                gpt_output = understand_GPT2(encoded_tensor, GPT_CONFIG)

            st.markdown("### GPT Output")
            st.write(gpt_output)
            st.markdown("### Output Shape")
            st.write(gpt_output.shape)

        st.markdown("---")
        st.markdown("## Summary")
        st.json(GPT_CONFIG)
        st.info(f"Input Text: {input_text}")
        st.info(f"Tokenized IDs: {encoded_tensor}")
        st.info(f"Input Embeddings Shape: {input_embeddings.shape}")
        if analysis_step3_cb:
            st.info(f"GPT Output Shape: {gpt_output.shape}")
    else:
        st.info("Please enter a sentence in the sidebar to start the demonstration.")


if __name__ == "__main__":
    main()
