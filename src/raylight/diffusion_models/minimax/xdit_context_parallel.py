import os

import torch

import comfy

from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift, AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP, PackedLayout, mask_row_values, pack_audio, patchify_video, rope_rotation_table, time_shift_sigma, unpack_audio, unpatchify_video
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

import raylight.distributed_modules.attention as xfuser_attn
from ..utils import pad_to_world_size


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _split_packed_sequence(h, rope_freqs, mod_segments):
    world_size = get_sequence_parallel_world_size()
    local_size = h.shape[0] // world_size
    start = get_sequence_parallel_rank() * local_size
    end = start + local_size
    local_segments = []
    for segment_start, segment_end, row in mod_segments:
        original_start = segment_start
        segment_start = max(segment_start, start)
        segment_end = min(segment_end, end)
        if segment_start < segment_end:
            if isinstance(row, torch.Tensor):
                row = row[segment_start - original_start:segment_end - original_start]
            local_segments.append((segment_start - start, segment_end - start, row))
    return h[start:end], rope_freqs[:, start:end], local_segments


# Volta and Turing have fp16 tensor cores but no bf16, so the model runs in fp32
# and attention dominates both activation memory and step time. The attention
# branch is safe in fp16 while the residual stays fp32, except for out_proj:
# its output peaks at 6.63e4 against fp16's 65504, so it is scaled by a power of
# two (exact, both projections have bias=False) and unscaled in fp32. The scale
# is applied in place on the fp16 tensor rather than on the fp32 value, so no
# full size fp32 copy of the attention output is materialised.
_ATTN_FP16_OUT_PROJ_SCALE = 64.0


def _attn_fp16():
    return os.environ.get("RAYLIGHT_ATTN_FP16") == "1"


def usp_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    fp16 = _attn_fp16() and x.dtype == torch.float32
    if fp16:
        x = x.to(torch.float16)
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    v = v.clone()
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = xfuser_optimized_attention(q, k, v, self.heads, skip_reshape=True, transformer_options=transformer_options)
    out = out.squeeze(0)
    if fp16:
        if out.dtype != torch.float16:
            out = out.to(torch.float16)
        out = out.div_(_ATTN_FP16_OUT_PROJ_SCALE)
        return self.out_proj(out).to(torch.float32).mul_(_ATTN_FP16_OUT_PROJ_SCALE)
    return self.out_proj(out)


# Off by default so output stays byte-identical to the single-pass path.
# Recommended value when activation memory is the constraint: 4096, which on a
# 16 GiB card freed 2.7 GiB for 1 second of wall clock on a 124 frame render.
# Enabling it changes which sample a seed produces (see the commit message).
def _mlp_chunk_tokens():
    try:
        return int(os.environ.get("RAYLIGHT_MLP_CHUNK_TOKENS", "0"))
    except ValueError:
        return 0


# fc2's output overflows fp16 by ~56x (measured peak 3.69e6 against 65504), so
# its input is scaled by a power of two before the projection and unscaled in
# fp32 afterwards. The swiglu itself is evaluated in fp32: it is pointwise and
# not exactly representable, and it is cheap relative to the projections.
_MLP_FP16_FC2_SCALE = 256.0


def _mlp_fp16():
    return os.environ.get("RAYLIGHT_MLP_FP16") == "1"


def _mlp_branch(self, x, fp16):
    gate, up = self.fc1(x).chunk(2, dim=-1)
    if not fp16:
        return self.fc2(torch.nn.functional.silu(gate).mul_(up))
    s = torch.nn.functional.silu(gate.to(torch.float32)).mul_(up.to(torch.float32))
    s = s.div_(_MLP_FP16_FC2_SCALE).to(torch.float16)
    return self.fc2(s).to(torch.float32).mul_(_MLP_FP16_FC2_SCALE)


def usp_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
    """DiT block that accumulates the residual in fp32 while the branches run in fp16.

    Loading the model in fp16 halves every activation, but the residual stream
    reaches ~1e7 across the 50 blocks, far past fp16's 65504. The branch outputs
    arrive already unscaled from usp_attn_forward and usp_mlp_forward.
    """
    attention = self.attn if attention is None else attention
    if x.dtype is not torch.float32:
        x = x.to(torch.float32)
    sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaln_proj(t_emb)
    h = _mod_scale_shift(self.norm1(x), sh_a, sc_a, mod_segments)
    att = attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    x = _mod_gate(x, g_a, att.to(torch.float32), mod_segments)
    h = _mod_scale_shift(self.norm2(x), sh_m, sc_m, mod_segments)
    m = self.mlp(h)
    return _mod_gate(x, g_m, m.to(torch.float32), mod_segments)


def usp_mlp_forward(self, x):
    # x is the packed sequence, [tokens, hidden]. Running it whole materializes
    # fc1's 2*ffn-wide output and the silu product across every token at once,
    # which dominates activation memory on long sequences. Slicing the token
    # dimension bounds both to the chunk while leaving the result identical.
    chunk = _mlp_chunk_tokens()
    fp16 = _mlp_fp16()
    if fp16 and x.dtype is torch.float32:
        x = x.to(torch.float16)
    if chunk <= 0 or x.shape[0] <= chunk:
        return _mlp_branch(self, x, fp16)

    out = None
    for start in range(0, x.shape[0], chunk):
        stop = start + chunk
        piece = _mlp_branch(self, x[start:stop], fp16)
        if out is None:
            out = x.new_empty((x.shape[0], piece.shape[-1]), dtype=piece.dtype)
        out[start:stop] = piece
    return out


def usp_dit_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype  # compute dtype


    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    # extra_conds prebuilds the layout once per sampling run
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=payload.get("keyframes"),
                              refs=payload.get("refs"))

    # model_base passes model_sampling.timestep(sigma) = sigma * 1000
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}

    # masked rows run at their own strength: mask value m puts a row at sigma = m * sigma_stream,
    # so its label is 1 - m * sigma, clamped at the cond timestep for fully preserved rows
    t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, AUDIO_COND_TIMESTEP)
    video_rows_t = None
    audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t

    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                      | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                      | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

    def rows_to_mod_index(rows_t, tag):
        # per-row timestep values -> per-row mod-row indices into the t_emb table
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],
                            dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    # embed
    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.zeros(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.zeros(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        # The refiner and condition_proj are unquantized, so on a card without
        # bf16 they are held in fp32 for the whole render while only being used
        # here. Keep them in fp16 and convert at the boundary.
        # condition_proj sees Qwen3-VL hidden states that project to ~96k, past
        # fp16's range, so it runs in fp32 regardless of the model dtype.
        text_states = self.token_refiner(self.condition_proj(text_states.to(torch.float32)),
                                         transformer_options=transformer_options).to(dtype)

    # segments are contiguous: assemble by slices, embed rows follow segment order
    # zeros, not empty: the segment loop below does not necessarily cover every
    # row (the sequence is padded to the world size), and an uninitialised row
    # is far more likely to hold a NaN bit pattern in fp16 than in fp32.
    h = torch.zeros(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:  # ref_audio / audio
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        # adaln projections consume interpolated coordinates of the time-embedding curve
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    # rotation table computed once per forward, consumed by the kitchen split-half rope
    rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
    # ===================== SP SPLIT ====================== #
    h, h_orig_size = pad_to_world_size(h, dim=0)
    rope_freqs, _ = pad_to_world_size(rope_freqs, dim=1)
    h, rope_freqs, mod_segments = _split_packed_sequence(h, rope_freqs, mod_segments)

    # blocks
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    # ===================== SP GATHER ===================== #
    h = get_sp_group().all_gather(h.contiguous(), dim=0)
    h = h[:h_orig_size]

    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
    if video_rows_t is not None:
        video_seg = (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)
    else:
        video_seg = (va, vb, t_row[seg_t["video"]])
    if audio_rows_t is not None:
        audio_seg = (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)
    else:
        audio_seg = (aa, ab, t_row[seg_t["audio"]])
    v, a = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a))

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)

    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
