"""Standalone native SAM3 image grounding runtime."""

import os
from importlib import resources

import torch
import torch.nn as nn
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.lora import wrap_mha_for_lora
from sam3.model.memory import CXBlock
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()

def _get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sam3_resource_path(relative_path: str) -> str:
    """Return a filesystem path for a resource bundled under the sam3 package."""
    return str(resources.files("sam3").joinpath(relative_path))


def _resolve_local_checkpoint_path():
    # Prefer the standalone application's checkpoint directory.
    repo_root = _get_repo_root()
    candidate_paths = [
        os.path.join(repo_root, "pretrained", "sam3.pt"),
        os.path.join(repo_root, "checkpoints", "sam3.pt"),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return None
    ## 新添代码


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder(
    enable_encoder_cross_attn_lora: bool = False,
    encoder_cross_attn_lora_r: int = 8,
    encoder_cross_attn_lora_alpha: int = 8,
    encoder_cross_attn_lora_dropout: float = 0.0,
) -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    cross_attention = MultiheadAttention(
        num_heads=8,
        dropout=0.1,
        embed_dim=256,
        batch_first=True,
    )
    if enable_encoder_cross_attn_lora:
        cross_attention = wrap_mha_for_lora(
            cross_attention,
            r=encoder_cross_attn_lora_r,
            alpha=encoder_cross_attn_lora_alpha,
            dropout=encoder_cross_attn_lora_dropout,
        )

    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=cross_attention,
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    pdc_branch,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "pdc_branch": pdc_branch,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    common_params["matcher"] = None
    model = Sam3Image(**common_params)

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(
    has_presence_token: bool = True,
    enable_encoder_cross_attn_lora: bool = False,
    encoder_cross_attn_lora_r: int = 8,
    encoder_cross_attn_lora_alpha: int = 8,
    encoder_cross_attn_lora_dropout: float = 0.0,
) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder(
        enable_encoder_cross_attn_lora=enable_encoder_cross_attn_lora,
        encoder_cross_attn_lora_r=encoder_cross_attn_lora_r,
        encoder_cross_attn_lora_alpha=encoder_cross_attn_lora_alpha,
        encoder_cross_attn_lora_dropout=encoder_cross_attn_lora_dropout,
    )
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if not sam3_image_ckpt:
        sam3_image_ckpt = ckpt
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Move the model to the requested device and enable evaluation mode."""
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    enable_inst_interactivity=False,
    enable_encoder_cross_attn_lora=False,
    encoder_cross_attn_lora_r=8,
    encoder_cross_attn_lora_alpha=8,
    encoder_cross_attn_lora_dropout=0.0,
    compile=False,
):
    """Build the standalone native SAM3 image grounding model."""
    if bpe_path is None:
        bpe_path = _sam3_resource_path("assets/bpe_simple_vocab_16e6.txt.gz")

    vision_encoder = _create_vision_backbone(
        compile_mode="default" if compile else None,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    text_encoder = _create_text_encoder(bpe_path)
    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer(
        enable_encoder_cross_attn_lora=enable_encoder_cross_attn_lora,
        encoder_cross_attn_lora_r=encoder_cross_attn_lora_r,
        encoder_cross_attn_lora_alpha=encoder_cross_attn_lora_alpha,
        encoder_cross_attn_lora_dropout=encoder_cross_attn_lora_dropout,
    )
    dot_prod_scoring = _create_dot_product_scoring()
    model = _create_sam3_model(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=None,
        pdc_branch=None,
        dot_prod_scoring=dot_prod_scoring,
        inst_interactive_predictor=None,
        eval_mode=eval_mode,
    )

    if checkpoint_path is None:
        checkpoint_path = _resolve_local_checkpoint_path()
    if checkpoint_path is None:
        raise FileNotFoundError(
            "SAM3 checkpoint not found. Run: python download_model.py"
        )
    _load_checkpoint(model, checkpoint_path)
    return _setup_device_and_mode(model, device, eval_mode)


__all__ = ["build_sam3_image_model"]
