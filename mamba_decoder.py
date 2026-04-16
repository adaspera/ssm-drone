# mamba_decoder.py
"""Mamba-based DETR decoder — drop-in replacement for RTDETRDecoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11
from ultralytics.nn.modules.transformer import MLP
from ultralytics.nn.modules.utils import _get_clones, bias_init_with_prob, inverse_sigmoid, linear_init


class MambaDecoderLayer(nn.Module):
    """Single Mamba decoder layer replacing DeformableTransformerDecoderLayer.

    Replaces:
    - Self-attention on queries  → Mamba1D on query sequence
    - MSDeformAttn cross-attention → bilinear sampling at reference points + Mamba1D
    - FFN → standard MLP (unchanged)
    """

    def __init__(self, d_model: int = 256, d_ffn: int = 1024, dropout: float = 0.0, n_levels: int = 3):
        """Initialize MambaDecoderLayer.

        Args:
            d_model: Hidden dimension.
            d_ffn: Feed-forward network dimension.
            dropout: Dropout probability.
            n_levels: Number of feature map levels for cross-sampling.
        """
        super().__init__()
        from mamba_ssm import Mamba

        # Self-Mamba: replaces self-attention on query sequence
        self.self_mamba = Mamba(d_model=d_model, d_state=64, d_conv=3, expand=1).to("cuda")
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # Cross-Mamba: replaces MSDeformAttn
        self.ctx_proj = nn.Linear(d_model * n_levels, d_model)
        self.cross_mamba = Mamba(d_model=d_model, d_state=64, d_conv=3, expand=1).to("cuda")
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.drop4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.n_levels = n_levels

    def _sample_at_reference(self, feats_spatial: list, refer_bbox: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample each feature level at reference box centers."""
        grid = refer_bbox[..., :2] * 2 - 1  # (bs, nq, 2)
        grid = grid.unsqueeze(2)  # (bs, nq, 1, 2)
        sampled = []
        for feat in feats_spatial:
            s = F.grid_sample(
                feat.float(), grid.float(), mode="bilinear", align_corners=False, padding_mode="border"
            )  # (bs, hd, nq, 1)
            sampled.append(s.squeeze(-1).permute(0, 2, 1).to(feat.dtype))  # (bs, nq, hd)
        return torch.cat(sampled, dim=-1)  # (bs, nq, hd * n_levels)

    def forward(self, embed, refer_bbox, feats_spatial, query_pos=None):
        device = embed.device
        dtype = embed.dtype
        mamba_device = next(self.self_mamba.parameters()).device

        # CPU guard for Ultralytics parameter-counting pass
        if not embed.is_cuda and mamba_device.type != "cpu":
            return embed

        # 1. Self-Mamba
        q = embed if query_pos is None else embed + query_pos
        tgt = self.self_mamba(q.to(mamba_device)).to(device=device, dtype=dtype)
        embed = self.norm1(embed + self.drop1(tgt))

        # 2. Cross-Mamba
        ctx = self._sample_at_reference(feats_spatial, refer_bbox)
        ctx = self.ctx_proj(ctx)
        tgt = self.cross_mamba((embed + ctx).to(mamba_device)).to(device=device, dtype=dtype)
        embed = self.norm2(embed + self.drop2(tgt))

        # 3. FFN
        tgt = self.linear2(self.drop3(self.act(self.linear1(embed))))
        embed = self.norm3(embed + self.drop4(tgt))
        return embed


class MambaDecoder(nn.Module):
    """Stack of MambaDecoderLayers with iterative bounding box refinement."""

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self, embed, refer_bbox, feats_spatial, bbox_head, score_head, pos_mlp, attn_mask=None):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats_spatial, query_pos=pos_mlp(refer_bbox))
            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class MambaDETRDecoderDetect(nn.Module):
    """Drop-in replacement for RTDETRDecoder using Mamba instead of transformer decoder layers.

    YAML usage (same as RTDETRDecoder):
        - [[P3, P4, P5], 1, MambaDETRDecoderDetect, [nc]]
    """

    export = False
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False
    _rtdetr_style = True  # signals parse_model to inject channels like RTDETRDecoder

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (256, 256, 256),
        hd: int = 256,
        nq: int = 300,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hd
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )

        decoder_layer = MambaDecoderLayer(d_model=hd, d_ffn=d_ffn, dropout=dropout, n_levels=self.nl)
        self.decoder = MambaDecoder(hd, decoder_layer, ndl, eval_idx)

        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    @staticmethod
    def _generate_anchors(shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list):
        feats_spatial = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats, shapes = [], []
        for proj in feats_spatial:
            h, w = proj.shape[2:]
            feats.append(proj.flatten(2).permute(0, 2, 1))
            shapes.append([h, w])
        feats = torch.cat(feats, 1)
        return feats, shapes, feats_spatial

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes
        features = self.enc_output(self.valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def forward(self, x: list, batch: dict | None = None):
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes, feats_spatial = self._get_encoder_input(x)

        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats_spatial,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        out = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return out
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, out)
