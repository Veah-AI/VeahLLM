"""
VEAH LLM - Solana-Native Language Model
Core model architecture and implementation
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    GenerationConfig
)
from typing import Optional, Dict, Any, List, Tuple
import json
from pathlib import Path


class VeahConfig(PretrainedConfig):
    """Configuration for VEAH LLM models"""

    model_type = "veah"

    def __init__(
        self,
        vocab_size=50432,  # Extended for Solana-specific tokens
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA for efficiency
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # Solana-specific parameters
        solana_vocab_size=5000,  # Additional Solana tokens
        enable_blockchain_attention=True,
        transaction_embedding_dim=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Solana-specific
        self.solana_vocab_size = solana_vocab_size
        self.enable_blockchain_attention = enable_blockchain_attention
        self.transaction_embedding_dim = transaction_embedding_dim


class SolanaAttentionLayer(nn.Module):
    """Custom attention layer for Solana-specific patterns"""

    def __init__(self, config: VeahConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Solana-specific: transaction pattern recognition
        self.tx_pattern_proj = nn.Linear(
            config.transaction_embedding_dim,
            self.hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        transaction_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]

        # Standard attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Add transaction pattern bias if available
        if transaction_embeddings is not None:
            tx_bias = self.tx_pattern_proj(transaction_embeddings)
            attn_output = attn_output + tx_bias

        return self.o_proj(attn_output)


class VeahLLM(PreTrainedModel):
    """Main VEAH LLM model class"""

    config_class = VeahConfig
    base_model_prefix = "veah"

    def __init__(self, config: VeahConfig):
        super().__init__(config)
        self.config = config

        # Initialize base transformer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Solana-specific embeddings
        self.solana_embed = nn.Embedding(
            config.solana_vocab_size,
            config.hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_layer(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def _create_layer(self, config):
        """Create a transformer layer with optional Solana attention"""
        if config.enable_blockchain_attention:
            return SolanaAttentionLayer(config)
        else:
            # Use standard transformer layer
            return nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.attention_dropout,
                activation=config.hidden_act,
                batch_first=True
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        solana_token_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

            # Add Solana-specific embeddings if provided
            if solana_token_ids is not None:
                solana_embeds = self.solana_embed(solana_token_ids)
                inputs_embeds = inputs_embeds + solana_embeds

        hidden_states = inputs_embeds

        # Pass through transformer layers
        for layer in self.layers:
            if isinstance(layer, SolanaAttentionLayer):
                hidden_states = layer(hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, src_mask=attention_mask)

        hidden_states = self.norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states if output_hidden_states else None,
        }

    @torch.no_grad()
    def generate_solana(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        context_type: str = "general",  # general, transaction, code, defi
        **kwargs
    ) -> str:
        """Generate text with Solana-specific context"""

        # Add context-specific prefixes
        context_prefixes = {
            "general": "",
            "transaction": "[TX_ANALYSIS]",
            "code": "[CODE_GEN]",
            "defi": "[DEFI_ANALYSIS]",
            "validator": "[VALIDATOR_INFO]",
        }

        prefix = context_prefixes.get(context_type, "")
        full_prompt = f"{prefix} {prompt}" if prefix else prompt

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_position_embeddings
        )

        # Generate
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs
        )

        outputs = self.generate(
            inputs.input_ids,
            generation_config=generation_config,
            attention_mask=inputs.attention_mask,
        )

        # Decode
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Remove prefix from response
        if prefix and response.startswith(prefix):
            response = response[len(prefix):].strip()

        return response


class VeahLLMForSolana:
    """High-level interface for VEAH LLM"""

    def __init__(self, model_name: str = "veah-7b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_pretrained(cls, model_name: str = "veah-7b", **kwargs):
        """Load pretrained VEAH model"""
        instance = cls(model_name)

        # Load model and tokenizer
        model_path = f"veah-ai/{model_name}"

        try:
            # Try loading from HuggingFace
            instance.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                **kwargs
            )
            instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            # Fallback to local loading
            instance.model = VeahLLM(VeahConfig())
            instance.tokenizer = instance._create_solana_tokenizer()

        instance.model.to(instance.device)
        instance.model.eval()

        return instance

    def _create_solana_tokenizer(self):
        """Create a tokenizer with Solana-specific vocabulary"""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Initialize tokenizer with BPE
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        # Add Solana-specific tokens
        solana_tokens = [
            "[LAMPORT]", "[SOL]", "[WSOL]", "[USDC]", "[USDT]",
            "[SLOT]", "[EPOCH]", "[BLOCKHASH]", "[SIGNATURE]",
            "[PUBKEY]", "[ACCOUNT]", "[PROGRAM]", "[INSTRUCTION]",
            "[TRANSACTION]", "[SYSVAR]", "[PDA]", "[RENT]",
            "[COMPUTE_UNIT]", "[PRIORITY_FEE]", "[MEV]",
            # Programs
            "[SYSTEM_PROGRAM]", "[TOKEN_PROGRAM]", "[ASSOCIATED_TOKEN]",
            "[METAPLEX]", "[RAYDIUM]", "[ORCA]", "[JUPITER]",
            "[MARINADE]", "[SERUM]", "[ANCHOR]",
            # Instructions
            "[TRANSFER]", "[CREATE_ACCOUNT]", "[CLOSE_ACCOUNT]",
            "[INITIALIZE]", "[SWAP]", "[ADD_LIQUIDITY]", "[STAKE]",
            # Special tokens
            "[TX_START]", "[TX_END]", "[CODE_START]", "[CODE_END]",
        ]

        # Would need actual training data to properly train the tokenizer
        # For now, returning a placeholder
        return tokenizer

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response for a prompt"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

    def analyze_transaction(self, tx_signature: str) -> Dict[str, Any]:
        """Analyze a Solana transaction"""
        prompt = f"""[TX_ANALYSIS] Analyze the following Solana transaction:
        Transaction Signature: {tx_signature}

        Provide a detailed analysis including:
        1. Transaction type and purpose
        2. Programs involved
        3. Token transfers and amounts
        4. Success/failure status
        5. Any notable patterns or risks
        """

        analysis = self.generate(prompt, temperature=0.3)

        return {
            "signature": tx_signature,
            "analysis": analysis,
            "timestamp": None,  # Would fetch from chain
            "programs": [],  # Would extract from analysis
        }

    def generate_code(
        self,
        description: str,
        language: str = "rust"
    ) -> str:
        """Generate Solana smart contract code"""

        lang_map = {
            "rust": "Rust/Anchor",
            "typescript": "TypeScript (client)",
            "python": "Python (client)",
        }

        prompt = f"""[CODE_GEN] Generate {lang_map.get(language, language)} code for Solana:
        Description: {description}

        Provide complete, production-ready code with proper error handling.
        """

        code = self.generate(prompt, temperature=0.5, max_length=1024)
        return code

    def query(self, question: str, context: str = "technical") -> str:
        """Answer Solana-related questions"""

        context_prompts = {
            "technical": "Provide a technical, detailed answer:",
            "eli5": "Explain in simple terms that anyone can understand:",
            "developer": "Answer from a developer's perspective with code examples:",
            "trader": "Answer focusing on trading and DeFi implications:",
        }

        prompt = f"""[SOLANA_QA] {context_prompts.get(context, '')}
        Question: {question}
        """

        return self.generate(prompt, temperature=0.6)