import tqdm
import torch
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv, fuse_linears
from awq.modules.fused.block import Qwen2MoeBlock
from awq.modules.fused.model import Qwen2MoeModel
from awq.modules.fused.moe import FusedSparseMoeBlock
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer as OldQwen2MoeDecoderLayer,
    Qwen2MoeForCausalLM as OldQwen2MoeForCausalLM,
)
from awq.modules.linear import WQLinear_GEMM
from awq.modules.fused.norm import FasterTransformerRMSNorm


class Qwen2MoeAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Qwen2MoeDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["gate"]

    @staticmethod
    def fuse_layers(model: OldQwen2MoeForCausalLM):
        fuser = Qwen2MoeFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldQwen2MoeForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldQwen2MoeDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldQwen2MoeForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldQwen2MoeDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # shared expert in
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.shared_expert.gate_proj, module.mlp.shared_expert.up_proj],
                inp=input_feat["mlp.shared_expert.gate_proj"],
                module2inspect=module.mlp.shared_expert,
            )
        )

        # shared expert out
        layers.append(
            dict(
                prev_op=module.mlp.shared_expert.up_proj,
                layers=[module.mlp.shared_expert.down_proj],
                inp=input_feat["mlp.shared_expert.down_proj"],
            )
        )

        # routed in
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[
                    w
                    for expert in module.mlp.experts
                    for w in [expert.gate_proj, expert.up_proj]
                ],
                inp=input_feat["mlp"],
                module2inspect=module.mlp,
            )
        )

        # routed out
        for i, expert in enumerate(module.block_sparse_moe.experts):
            layers.append(
                dict(
                    prev_op=expert.up_proj,
                    layers=[expert.down_proj],
                    inp=input_feat[f"mlp.experts.{i}.down_proj"],
                )
            )

        return layers


class Qwen2MoeFuser:
    def __init__(self, model: OldQwen2MoeForCausalLM):
        self.model = model

        self.qwen2_blocks: List[Tuple[str, OldQwen2MoeDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Qwen2MoeDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldQwen2MoeDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon,
            )

            sparse_moe = module.mlp
            if isinstance(sparse_moe.experts[0].gate_proj, WQLinear_GEMM):
                fused_w1w3s = [
                    fuse_linears(
                        [
                            sparse_moe.experts[i].gate_proj,
                            sparse_moe.experts[i].up_proj,
                        ],
                        device,
                    )
                    for i in range(len(sparse_moe.experts))
                ]

                stacked_w1w3s = fuse_linears(
                    fused_w1w3s, device, dim=0, operation=torch.stack
                )

                stacked_w2s = fuse_linears(
                    [expert.down_proj for expert in sparse_moe.experts],
                    device,
                    dim=0,
                    operation=torch.stack,
                )

                sparse_moe = FusedSparseMoeBlock(
                    top_k=sparse_moe.top_k,
                    gate=sparse_moe.gate,
                    ws=stacked_w1w3s,
                    w2s=stacked_w2s,
                )

            blocks.append(
                Qwen2MoeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    moe=sparse_moe,
                    shared_expert=module.mlp.shared_expert,
                    shared_expert_gate=module.mlp.shared_expert_gate,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        model_norm = FasterTransformerRMSNorm(
            self.model.model.norm.weight,
            self.model.model.norm.variance_epsilon,
        )

        self.model.model = Qwen2MoeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            model_norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
