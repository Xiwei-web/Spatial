# 模型清单（MODEL_INVENTORY）

由 `scripts/build_inventory.py` 从 `configs/models/*.yaml` 生成，请勿手改。
每个字段的出处见对应 yaml 的 `sources`；查不到的字段为 “—”，并列在 yaml 的 `unknown` 里。

## 总表

| 模型 | HF repo @ revision | 体积 GB | 许可 | LM 层/维度/头 | 逐层混合机制 | 视觉 token/图 | 角色（基座） | 环境 |
|---|---|---|---|---|---|---|---|---|
| Cambrian-S-7B | nyu-visionx/Cambrian-S-7B @ `556e72fd` | 32.1 | apache-2.0 | 28/3584/28 | 全部 full_attention（28 层） | — | spatial_finetune（Qwen/Qwen2.5-7B-Instruct (README front …） | 未适配 |
| GLM-4.6V-Flash | zai-org/GLM-4.6V-Flash @ `411bb4d7` | 20.6 | mit | 40/4096/32 | 全部 full_attention（40 层） | 可变：n = grid_t*grid_h*grid_w / merge_size^2 = (H'/28) * (W'/… | general | tf5 |
| InternVL3_5-8B | OpenGVLab/InternVL3_5-8B @ `9bb6a56a` | 17.1 | apache-2.0 | 36/4096/32 | 全部 full_attention（36 层） | 256 per 448x448 tile: num_image_token = int((448 // 14) ** … | general（Not a spatial fine-tune. README.md fron…） | tf4 |
| LLaVA-OneVision-7B | llava-hf/llava-onevision-qwen2-7b-ov-hf @ `0d506805` | 16.1 | apache-2.0 | 28/3584/28 | 全部 full_attention（28 层） | Single image in a sample (default AnyRes path): N = 729 (ba… | general | tf5 |
| MiniCPM-V-4_5 | openbmb/MiniCPM-V-4_5 @ `14a20049` | 17.4 | apache-2.0 | 36/4096/32 | 全部 full_attention（36 层） | 64 * (1 + n_slices)。每个视图固定 64 个占位（image_feature_size=64 个 <… | general | tf4 |
| Qwen3.5-2B | Qwen/Qwen3.5-2B @ `15852e8c` | 4.6 | apache-2.0 | 24/2048/8 | linear_attention×18；full_attention×6（层 3,7,11,15,19,23） | N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, … | general | tf5 |
| Qwen3.5-4B | Qwen/Qwen3.5-4B @ `851bf6e8` | 9.3 | apache-2.0 | 32/2560/16 | linear_attention×24；full_attention×8（层 3,7,11,15,19,23,27,31） | N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, … | general | tf5 |
| Qwen3.5-9B | Qwen/Qwen3.5-9B @ `c2022362` | 19.3 | apache-2.0 | 32/4096/16 | linear_attention×24；full_attention×8（层 3,7,11,15,19,23,27,31） | N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, … | general | tf5 |
| Qwen3.6-27B | Qwen/Qwen3.6-27B @ `6a9e13bd` | 55.6 | apache-2.0 | 64/5120/24 | linear_attention×48；full_attention×16（层 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63） | N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, … | general | tf5 |
| Qwen3.6-35B-A3B | Qwen/Qwen3.6-35B-A3B @ `995ad96e` | 71.9 | apache-2.0 | 40/2048/16 | linear_attention×30；full_attention×10（层 3,7,11,15,19,23,27,31,35,39） | N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, … | general | tf5 |
| Qwen3.8-27B | Qwen/Qwen3.8-27B @ `1d4bf0f2` | 55.6 | apache-2.0 | 64/5120/24 | linear_attention×48；full_attention×16（层 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63） | N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, … | general | tf5 |
| RoboBrain2.0-7B | BAAI/RoboBrain2.0-7B @ `f844fdfe` | 16.6 | apache-2.0 | 28/3584/28 | 全部 full_attention（28 层） | (H'/28) * (W'/28), where (H', W') is the smart_resize outpu… | spatial_finetune | tf5 |
| SenseNova-SI-1.1-InternVL3-2B | sensenova/SenseNova-SI-1.1-InternVL3-2B @ `81c0e4c5` | 4.2 | apache-2.0 | 28/1536/12 | 全部 full_attention（28 层） | 256 per 448x448 tile: num_image_token = int((448 // 14) ** … | spatial_finetune（OpenGVLab/InternVL3-2B - from README.md…） | tf4 |
| SenseNova-SI-1.2-InternVL3-8B | sensenova/SenseNova-SI-1.2-InternVL3-8B @ `57bfba82` | 15.9 | apache-2.0 | 28/3584/28 | 全部 full_attention（28 层） | 256 per 448x448 tile: num_image_token = int((448 // 14) ** … | spatial_finetune（OpenGVLab/InternVL3-8B - from README.md…） | tf4 |
| SenseNova-SI-1.3-InternVL3-8B | sensenova/SenseNova-SI-1.3-InternVL3-8B @ `2081462c` | 15.9 | apache-2.0 | 28/3584/28 | 全部 full_attention（28 层） | 256 per 448x448 tile: num_image_token = int((448 // 14) ** … | spatial_finetune（OpenGVLab/InternVL3-8B - from README.md…） | tf4 |
| VST-7B-RL | rayruiyang/VST-7B-RL @ `b18ec54b` | 16.6 | apache-2.0 | 28/3584/28 | 全部 full_attention（28 层） | (H'/28) * (W'/28), where (H', W') is the smart_resize outpu… | spatial_finetune（Qwen2.5-VL 7B (exact hub repo not state…） | tf5 |
| gemma-4-31B-it | google/gemma-4-31B-it @ `842da379` | 62.6 | apache-2.0 | 60/5376/32 | sliding_attention×50；full_attention×10（层 5,11,17,23,29,35,41,47,53,59） | 可变，不是固定 280。n = (H'/48) * (W'/48)，H'、W' 为缩放后的高宽（48 的倍数），n <… | general | tf5 |

## 逐模型明细

### Cambrian-S-7B

- **hf_repo**: nyu-visionx/Cambrian-S-7B
- **revision**: 556e72fded09352f6aa4e9c100c9f39e6fe0fba7
- **revision_source**: MANIFEST.md table row (line 17) and reproduce command (line 41)
- **size_bytes**: 32126660771
- **license**: apache-2.0
- **license_source**: README.md front matter line 2 (MANIFEST.md also lists apache-2.0)
- **role**: spatial_finetune
- **base_model**: Qwen/Qwen2.5-7B-Instruct (README front matter "base_model", line 3; config.json _name_or_path). Note this base is a text-only LLM, not a VLM: the card describes "Qwen2.5-7B-Instruct + SigLIP2-SO400M vision encoder + 2-layer MLP adapter" (README:151) trained with a "4-stage pipeline (image alignment -> image IT -> video IT -> spatial IT)" (README:154) on VSI-590K (spatial reasoning) + general video instruction data (README:155). Card description: "Cambrian-S-7B is a spatially-grounded multimodal large language model that excels at spatial reasoning in video understanding." (README:147)

**loader**

- model_class: CambrianQwenForCausalLM
- processor_class: —
- trust_remote_code: —
- config_transformers_version: 4.39.2
- notes: model_type "cambrian_qwen" has no entry in the transformers 5.5.4 auto mappings, so AutoModel/AutoProcessor cannot load it. Official code: https://github.com/cambrian-mllm/cambrian-s (README:143). Card usage: from cambrian.model.builder import load_pretrained_model; tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "cambrian-s-7b", device_map="cuda"); conv_templates["qwen_2"]; prompt "<image>\nWhat objects are in this scene?"; model.generate(input_ids, images=image_tensor, image_sizes=image_sizes) (README:160-176). The card states no transformers / torch version; config.json was written by transformers 4.39.2 and the code's compatibility with transformers 5.5.4 is unknown. Weights are float32 on disk (config torch_dtype float32, 7 shards, index total_size 32121395328). The SigLIP2 vision tower weights ARE included in this repo's safetensors (421 tensors under model.vision_tower_aux_list.0.vision_tower.vision_model.*, encoder layers 0..25, plus post_layernorm); whether the official loader nevertheless first instantiates google/siglip2-so400m-patch14-384 from the Hub (config/processor download) cannot be told without the code. Tokenizer files: vocab.json…

**language_model**

- model_type: cambrian_qwen
- num_layers: 28
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: config.json: use_sliding_window false, sliding_window 131072, max_window_layers 28. Under the transformers Qwen2 rule (sliding only if use_sliding_window and layer index >= max_window_layers) this is full self-attention in all 28 layers. CAVEAT: the actual attention code is the official CambrianQwen implementation, which was not available to read; the list above assumes it keeps standard Qwen2 attention. rope_theta 1e6, max_position_embeddings 32768.
- mlp_type: dense
- head_level_analysis: By config, all 28 layers are standard attention with 28 query heads / 4 kv heads, same as Qwen2. Hook points and whether the official model class alters the forward pass (e.g. how visual tokens are spliced) must be checked against the official code.

**vision_encoder**

- type: SigLIP2 so400m patch14 384 (config mm_vision_tower_aux_list ["google/siglip2-so400m-patch14-384"]; card "SigLIP2-SO400M vision encoder", "SigLIP-384")
- num_layers: 26
- hidden_size: 1152
- patch_size: 14
- resolution_strategy: 384x384 per view (from the tower name; 729 = 27x27 tokens per view per mm_vision_tower_aux_token_len_list). config: image_aspect_ratio "anyres", anyres_max_subimages 9, si_token_len 729, miv_token_len 64, video_max_frames 128, max_images_per_sample 128. The actual resize/tiling procedure is in the official code (no preprocessor config is shipped).
- position_encoding: Learned absolute position embedding table (tensor vision_model.embeddings.position_embedding.weight exists in the weight index; shape not read). Feature selection per config: mm_vision_select_layer -2, mm_vision_select_feature "patch".

**connector**

- type: mlp2x_gelu (mm_projector_type); weight index has model.mm_projector.0 and model.mm_projector.2 (two Linear layers), mm_hidden_size 1152 -> LM hidden 3584; plus model.image_newline parameter
- token_merge: —
- tokens_per_image: —
- token_to_region: —

**multi_view**

- default: Per config: image_aspect_ratio "anyres" with anyres_max_subimages 9 and mm_use_im_newline_token true, i.e. multi-sub-image tiling is the configured default for single images; details (grid selection, whether a global view is added, ordering) are not in local files.
- force_single_view: —

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md searched for grounding / bounding box / bbox / point / coordinate - no match (the phrase "spatially-grounded" at README:147 is a description, not a grounding interface). The tokenizer inherits Qwen2.5 special tokens <|box_start|>/<|box_end|> (added_tokens.json) but the card does not say they are used.

**thinking**

- has_thinking_mode: False
- how_to_disable: —
- source: README.md does not mention thinking/reasoning modes; the tokenizer chat template (tokenizer_config.json) has no thinking branch. The official conversation template "qwen_2" (README:169) was not available to read.

**查不到 / 未确认**

- Everything implemented in the official code (https://github.com/cambrian-mllm/cambrian-s), which is not present locally: model class CambrianQwenForCausalLM, image preprocessing, anyres tiling rule, total tokens per image, token order, token -> region mapping, how to force a single view, the 'qwen_2' conversation template.
- image_position: 14 (config.json:16): meaning not documented in the card; not interpreted here.
- miv_token_len 64: how 729 tokens are reduced to 64 for multi-image/video input is not in local files.
- mm_vision_select_layer -2 vs. 26 encoder layers stored: which SigLIP2 layer's output is used (and whether the stored tower is already truncated) needs the official code.
- connector_only: true and unfreeze_mm_vision_tower: false (config.json): training-time flags whose inference effect is not documented.
- Whether loading needs network access to google/siglip2-so400m-patch14-384 (processor/config) even though tower weights are in the checkpoint.
- Required transformers/torch versions for the official code (card states none; config written by 4.39.2).
- language_model.layer_mixers assumes standard Qwen2 attention in the custom class (config-based, not code-verified).
- bbox.*: no grounding prompt or coordinate convention in the card.

### GLM-4.6V-Flash

- **hf_repo**: zai-org/GLM-4.6V-Flash
- **revision**: 411bb4d77144a3f03accbf4b780f5acb8b7cde4e
- **revision_source**: .cache/huggingface/download/*.metadata 第 1 行（14 个文件一致）。MANIFEST.md 未收录该模型，_logs/GLM-4.6V-Flash_download.log 里没有 repo id / sha。同一字段在 gemma-4-31B-it 上与 MANIFEST 的 sha 一致，据此认定其为 commit sha。
- **size_bytes**: 20605705573
- **license**: mit
- **license_source**: README.md:6 front matter `license: mit`
- **role**: general
- **base_model**: —

**loader**

- model_class: Glm4vForConditionalGeneration
- processor_class: Glm46VProcessor
- trust_remote_code: False
- config_transformers_version: 5.0.0rc0
- notes: README.md:74 `pip install transformers>=5.0.0rc0`（写在 vLLM 安装段，另有 vllm>=0.12.0、sglang>=0.5.6.post1）。 注意 config.json 的 model_type=glm4v / architectures=Glm4vForConditionalGeneration， 而 preprocessor_config.json 指定 Glm46VImageProcessor / Glm46VProcessor（video 为 Glm46VVideoProcessor）：模型类与处理器类分属两个模块。 tokenizer_config.json 的 tokenizer_class=PreTrainedTokenizer。 README 示例在 generate 前 `inputs.pop("token_type_ids", None)`（README.md:112），torch_dtype="auto", device_map="auto"。 generation_config.json（transformers_version 4.57.1）：do_sample=true, top_p=0.6, top_k=2, temperature=0.8；README 另建议 repetition_penalty=1.1。

**language_model**

- model_type: glm4v_text
- num_layers: 40
- hidden_size: 4096
- num_attention_heads: 32
- num_key_value_heads: 2
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: config 中没有 layer_types / sliding_window 字段；Glm4vTextModel.forward 只构造一个 create_causal_mask 并用于全部层（modeling_glm4v.py:903），所有 40 层都是标准因果 softmax 自注意力（Glm4vTextAttention :510-577，is_causal=True）。 GQA：32 个 q 头、num_key_value_heads=2，即每 16 个 q 头共享一组 K/V（num_key_value_groups = 32 // 2，:525）。 q/k/v_proj 带 bias（attention_bias=true），o_proj 无 bias。 位置编码为 3D M-RoPE：mrope_section=[8,12,12]（时间/高/宽），rope_theta=500000，partial_rotary_factor=0.5； 图像 token 的 (t,h,w) 位置由 get_vision_position_ids 生成（:949-1003）。
- mlp_type: dense（Glm4vTextMLP，intermediate_size=13696，hidden_act silu；本模型不是 glm4v_moe）
- head_level_analysis: 40 层都可做 head 级分析与注意力屏蔽（全局因果 softmax 注意力，o_proj 输入为 32 头 x 128 维拼接）。 注意 K/V 只有 2 组：同组 16 个头共用同一份 K 和 V，头之间只有 Q（及 o_proj 切片）不同， 对 K/V 的干预无法做到单头粒度。attn_weights 的返回取决于注意力后端（:560-573 通过 ALL_ATTENTION_FUNCTIONS 选择， 缺省回落 eager_attention_forward）；非 eager 后端是否返回权重未核实。

**vision_encoder**

- type: Glm4vVisionModel (model_type glm4v_vision)
- num_layers: 24
- hidden_size: 1536
- patch_size: 14
- resolution_strategy: dynamic（原生分辨率，单视图）。Glm46VImageProcessor 用 smart_resize（image_processing_glm46v.py:50-83, :152-161）： 高宽各自四舍五入到 28（=patch_size*merge_size）的倍数；单张图按 num_frames=temporal_patch_size=2 计入， 若 2*H'*W' > max_pixels 则等比缩小，若 2*H'*W' < min_pixels 则等比放大；长宽比上限 200。 本模型 preprocessor_config.json：size.shortest_edge=12544（min_pixels），size.longest_edge=9633792（max_pixels）。 由此推出（未实测）单图 H'*W' 上限 4816896 像素、下限 6272 像素。 单张图在送入 Conv3d 前被复制成 2 帧（:184-188），grid_t=1。归一化用 CLIP mean/std。 vision_config.image_size=336 只决定可学习位置表的大小（24x24），不是输入分辨率。
- position_encoding: 两种叠加：(1) 可学习绝对位置表 nn.Embedding((336/14)^2=576, hidden)，视作 24x24 网格， 按各 patch 的归一化坐标用 F.grid_sample（bicubic, align_corners=False, padding_mode border）插值到实际网格后相加 （Glm4vVisionEmbeddings，modeling_glm4v.py:136-207）；(2) 每个视觉注意力层内的 2D RoPE（h、w 位置， Glm4vVisionRotaryEmbedding(head_dim//2)，rot_pos_emb :711, :730-757）。

**connector**

- type: post_layernorm → Conv2d downsample（kernel=stride=spatial_merge_size=2，1536→4096）→ Glm4vVisionPatchMerger（Linear+LayerNorm+GELU 后接门控 MLP，输出 4096 维）（modeling_glm4v.py:119-133, :714-724, :806-814）
- token_merge: 2x2 空间合并（4 个 14px patch → 1 个 token，4 倍压缩）；另有时间方向 2 帧合 1（单图为同一帧复制）
- tokens_per_image: 可变：n = grid_t*grid_h*grid_w / merge_size^2 = (H'/28) * (W'/28)，H'、W' 为 smart_resize 后的高宽 （processing_glm46v.py:115-120；image_processing_glm46v.py:190-192）。 按 preprocessor_config 的像素范围推算（未实测）：最少 8 个、最多 6144 个 token。 模板把每张图写成 <|begin_of_image|><|image|><|end_of_image|>，processor 把 <|image|> 展开成 n 个。
- token_to_region: 可行，规则确定。图像处理器把 patch 按“合并后网格行优先、每个 2x2 块内 4 个 patch 相邻”的顺序排列 （permute 为 (B, grid_t, gh, gw, mh, mw, ...)，image_processing_glm46v.py:194-213）， 视觉塔末端每 4 个相邻 patch 经 Conv2d 合成 1 个 token（modeling_glm4v.py:808-812）， LLM 侧位置 id 也是高度 repeat_interleave、宽度 repeat（:993-998），即行优先。 第 i 个图像 token ↔ row = i // (W'/28), col = i % (W'/28)，对应缩放后图像上的 28x28 像素格 [col*28, (col+1)*28) x [row*28, (row+1)*28)；再按 (原宽/W', 原高/H') 映回原图。 每个 token 经过 24 层视觉注意力，内容上会混入其他位置的信息。

**multi_view**

- default: 否。Glm46VImageProcessor 只做一次 smart_resize，没有切片、多裁剪或缩略图；每张图恒为单视图，token 数随分辨率变化。
- force_single_view: 无需设置（本来就是单视图）。要固定 token 数：把输入图预先缩放到固定的 28 倍数尺寸，或通过 images_kwargs 的 size={shortest_edge, longest_edge}（即 min/max pixels）收紧范围。

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md 全文没有 grounding / bounding box 的提示词、输出格式或坐标约定（只让读者去 GitHub zai-org/GLM-V）。tokenizer_config.json 中存在特殊 token <|begin_of_box|> 与 <|end_of_box|>，但本地任何文件都没有说明其用法。

**thinking**

- has_thinking_mode: True
- how_to_disable: apply_chat_template(..., enable_thinking=False)。模板行为（chat_template.jinja）：当 enable_thinking 已定义且为假时， 在每条 user 消息末尾追加 `/nothink`（:61，若内容已以 /nothink 结尾则不重复）， 并在 generation prompt 的 `<|assistant|>` 换行后预填 `<think></think>\n`（:138-139）。 不传 enable_thinking 时两处都不触发，即默认不关闭思考。历史 assistant 消息统一写成 `<think></think>` 或带 reasoning_content 的 `<think>...</think>`（:74-78）。 README 未说明思考开关，只提到模型“may still overthink or even repeat itself”（README.md:139）。
- source: chat_template.jinja:61,74-78,138-139；README.md:134-139

**查不到 / 未确认**

- bbox 全部字段：README 没有 grounding 提示词/格式/坐标约定；<|begin_of_box|>/<|end_of_box|> token 存在但用法未在本地文件中说明（README 指向 GitHub，未联网查）。
- revision 未出现在 MANIFEST.md 或下载日志中，只能取自 .cache 的下载元数据。
- 不传 enable_thinking 时模型是否一定输出 <think>：模板只是不加 /nothink，实际行为未实测。
- tokens_per_image 的上下限（8 / 6144）与像素上下限是按 smart_resize 公式手算，未用真实图像跑 processor 验证。
- model_type=glm4v 的模型类与 Glm46V 处理器组合在 transformers 5.5.4 下能否直接加载：未加载验证。
- 非 eager 注意力后端是否返回 attn_weights：未核实。

### InternVL3_5-8B

- **hf_repo**: OpenGVLab/InternVL3_5-8B
- **revision**: 9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79
- **revision_source**: MANIFEST.md (model table row InternVL3_5-8B, and the pinned `hf download ... --revision` command)
- **size_bytes**: 17072800269
- **license**: apache-2.0
- **license_source**: README.md front matter line 2 (`license: apache-2.0`), README.md line 818, MANIFEST.md table
- **role**: general
- **base_model**: Not a spatial fine-tune. README.md front matter (lines 5-7) declares `base_model: - OpenGVLab/InternVL3_5-8B-MPO` with `base_model_relation: finetune`. README.md line 135: "We initialize the language model using the Qwen3 series and GPT-OSS, and the vision encoder using InternViT-300M and InternViT-6B." config.json llm_config._name_or_path (line 15) is a training-machine path whose last component is "Qwen3-8B"; llm_config.architectures is Qwen3ForCausalLM.

**loader**

- model_class: AutoModel (config.json auto_map -> modeling_internvl_chat.InternVLChatModel; architectures: InternVLChatModel)
- processor_class: For the remote-code path (README): AutoTokenizer(path, trust_remote_code=True, use_fast=False) -> tokenizer_class Qwen2Tokenizer; images are preprocessed by the caller with the README functions build_transform / dynamic_preprocess / load_image (no processor object). The directory additionally ships processor_config.json / preprocessor_config.json / video_preprocessor_config.json / chat_template.jinja that name processor_class InternVLProcessor (image_processor_type GotOcr2ImageProcessorFast, video_processor_type InternVLVideoProcessor); see loader.notes.
- trust_remote_code: True
- config_transformers_version: —
- notes: - config.json top-level `transformers_version` is null (line 58) and llm_config has no transformers_version; generation_config.json records "4.52.1". - README.md line 360: "Please use transformers>=4.52.1 to ensure the model works normally. For the 20B version of our model, transformers>=4.55.0 is required." - Remote code asserts transformers >= 4.37.0 (modeling_internvl_chat.py:52) but imports `Qwen3ForCausalLM, Qwen3MoeForCausalLM` from transformers at module import time (line 18), so a transformers that has Qwen3 is required in practice. - README loading call: AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda() - This is the "GitHub format" checkpoint (README.md line 41: "we provide two model formats: the GitHub format ..., consistent with prior releases, and the HF format ..., aligned with the official Transformers standard"). config.json model_type is `internvl_chat` with auto_map; weight key prefixes are language_model.* / vision_model.* / mlp1.* (model.safetensors.index.json). The HF-native class in transformers uses model_type `internvl` (transformers models/internvl/configuration_in…

**language_model**

- model_type: qwen3
- num_layers: 36
- hidden_size: 4096
- num_attention_heads: 32
- num_key_value_heads: 8
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: All 36 layers are standard causal full self-attention with grouped-query attention (32 query heads, 8 KV heads, head_dim 128 stated in config.json llm_config). No sliding window is active: llm_config has use_sliding_window: false and sliding_window: null (max_window_layers: 36 is therefore inert). In transformers 5.5.4 Qwen3Config.__post_init__ sets sliding_window=None when use_sliding_window is false and then builds layer_types as all "full_attention" (models/qwen3/configuration_qwen3.py:87-94). architectures: Qwen3ForCausalLM (not the Moe class).
- mlp_type: dense (architectures = Qwen3ForCausalLM; hidden_act silu, intermediate_size 12288)
- head_level_analysis: All 36 layers have the same structure (32 heads), so head-level analysis is structurally possible on every layer. Precondition: the LLM must run with the eager attention backend, otherwise attention weights are not materialised. The remote code forces llm_config._attn_implementation to 'flash_attention_2' when use_flash_attn=True and flash_attn is importable (modeling_internvl_chat.py:61-63); load with use_flash_attn=False to get 'eager'. In transformers 5.5.4 only eager_attention_forward returns the softmaxed weights (models/qwen3/modeling_qwen3.py:196-218) and config.output_attentions raises unless _attn_implementation is 'eager' (configuration_utils.py:346-351). InternVLChatModel.forward passes output_attentions through to the language model and returns outputs.attentions (modeling_internvl_chat.py:148,176). Head masking is not offered by an argument in the code read; it would need hooks. Not tested (no weights were loaded). Vision encoder: InternAttention never returns attention weights (_naive_attn returns only x; FlashAttention asserts not need_weights), so ViT attention needs hooks / re-computation.

**vision_encoder**

- type: InternVisionModel from the shipped modeling_intern_vit.py. config.json vision_config.model_type is the string "intern_vit_6b", but the actual dimensions are hidden_size 1024, 24 layers, 16 heads, intermediate_size 4096, norm_type layer_norm, qkv_bias true. README naming: line 135 says the series uses "InternViT-300M and InternViT-6B"; the family table (line 54) lists InternVL3.5-8B with #Vision Param 0.3B. The README does not print the exact vision checkpoint name for the 8B model.
- num_layers: 24
- hidden_size: 1024
- patch_size: 14
- resolution_strategy: Encoder input is fixed 448x448 per tile (vision_config.image_size 448, force_image_size 448) -> 32x32 = 1024 patch tokens + 1 CLS. Higher resolution is handled outside the encoder by dynamic tiling: config dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true (see multi_view).
- position_encoding: Learned absolute position embedding: nn.Parameter of shape (1, 1025, 1024) = CLS + 32x32 (modeling_intern_vit.py:149-152). At forward time the patch part is passed through _get_pos_embed, which bicubic-interpolates (F.interpolate, mode='bicubic', align_corners=False) the 32x32 grid to the actual (H, W) patch grid (lines 154-173); at 448x448 input this is the identity size. No rotary / 2D relative encoding in the ViT.

**connector**

- type: pixel_shuffle (space-to-depth, downsample_ratio 0.5, ps_version v2) followed by `mlp1` = Sequential(LayerNorm(4096), Linear(4096, 4096), GELU(), Linear(4096, 4096)) (modeling_internvl_chat.py:89-94; 4096 = vit_hidden 1024 * int(1/0.5)^2; llm hidden 4096). Uses the last ViT layer (select_layer -1), CLS token dropped.
- token_merge: 2x2 patch merge by pixel_shuffle (4 ViT patch tokens -> 1 LLM token, channels concatenated 1024 -> 4096); 4x token reduction
- tokens_per_image: 256 per 448x448 tile: num_image_token = int((448 // 14) ** 2 * (0.5 ** 2)) = int(1024 * 0.25) = 256 (modeling_internvl_chat.py:58). README.md line 141 agrees ("1024 visual tokens ... compressed into 256 tokens via a pixel shuffle module"); processor_config.json image_seq_length = 256. Per image: 256 * num_tiles, where num_tiles = grid tiles (1..max_num) + 1 thumbnail if grid tiles > 1. In the prompt each image becomes '<img>' + '<IMG_CONTEXT>' * 256 * num_tiles + '</img>' (modeling_internvl_chat.py:295), i.e. one <img>...</img> pair per image, not per tile. Token ids (added_tokens.json): <img> 151669, </img> 151670, <IMG_CONTEXT> 151671.
- token_to_region: Derived from the shipped remote code (extract_feature + pixel_shuffle, ps_version=v2), not from a doc statement. 1) The ViT output drops the CLS token (vit_embeds[:, 1:, :]) and is reshaped to (tiles, 32, 32, 1024), row-major: dim1 = patch row (y), dim2 = patch column (x). 2) pixel_shuffle(scale_factor=0.5): view -> merges horizontally adjacent patch pairs into channels; permute(0,2,1,3); view -> merges vertically adjacent pairs; for ps_version != 'v1' a final permute(0,2,1,3) restores (row, col) order. Result (tiles, 16, 16, 4096). 3) reshape(tiles, -1, 4096) flattens row-major, so within one 448x448 tile, token k (0..255) has row i = k // 16, col j = k % 16 and covers ViT patches rows 2i..2i+1, cols 2j..2j+1, i.e. tile pixels x in [28*j, 28*j+28), y in [28*i, 28*i+28). Channel concat order inside a token: patch(2i,2j), patch(2i,2j+1), patch(2i+1,2j), patch(2i+1,2j+1). 4) Across tiles: the n-th <IMG_CONTEXT> position of an image (n counted from 0) maps to tile t = n // 256, k = n % 256, because vit_embeds.reshape(-1, C) is written into the IMG_CONTEXT positions in order. Tile order = order of pixel_values along dim 0 (see multi_view). 5) Tile pixel -> original image pixel depends…

**multi_view**

- default: Dynamic tiling ON by default in the README pipeline: load_image(image_file, input_size=448, max_num=12) calls dynamic_preprocess(..., use_thumbnail=True, max_num=max_num); config.json has dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true. IMPORTANT: modeling_internvl_chat.py (inference code) never reads dynamic_image_size / min_dynamic_patch / max_dynamic_patch / use_thumbnail; tiling is done entirely by the caller's preprocessing, and the model just consumes pixel_values of shape (total_tiles, 3, 448, 448) plus num_patches_list. Tile choice: the (cols, rows) grid with cols*rows <= max_num whose aspect ratio is closest to the image's; the image is resized to (448*cols, 448*rows) (aspect ratio not preserved exactly) and cropped. Tile order in the sequence: row-major over the grid - tile i has column i % cols and row i // cols (left-to-right, then top-to-bottom). The thumbnail (whole image resized to 448x448) is appended LAST, and only when there is more than one grid tile. All tiles of one image share a single <img> ... </img> span of 256*num_tiles <IMG_CONTEXT> tokens, in the same order as pixel_values. Multi-image: pixel_values of the images a…
- force_single_view: Call the README loader with max_num=1: load_image(path, max_num=1). Then target_ratios = {(1, 1)}, blocks = 1, and the thumbnail is NOT added because of `if use_thumbnail and len(processed_images) != 1:` (README.md line 501). Result: pixel_values of shape (1, 3, 448, 448) = the whole image resized to 448x448 (aspect ratio not preserved), 256 image tokens. Equivalent without the helper: pass pixel_values with exactly one 448x448 tensor per image and num_patches_list=[1] (chat() defaults num_patches_list to [pixel_values.shape[0]], modeling_internvl_chat.py:271-272). Note a (1,1) grid can also be selected naturally for near-square images under max_num=12 only if it is the closest aspect ratio and wins the tie-break; do not rely on that - set max_num=1.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated in the local model card. README.md has a "### Visual Grounding" heading (line 311) that contains only a results image; a search of README.md for <ref>, <box>, "bounding", "coordinate", "1000" found no grounding prompt, output format or normalisation. The tokenizer does define <ref> </ref> <box> </box> <quad> </quad> as added tokens (added_tokens.json ids 151672-151677), which shows the vocabulary has grounding markers but does not document the convention.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is OFF by default and is switched on only by a system prompt. README.md lines 410-426: "To enable thinking mode, please set the system prompt to our Thinking System Prompt. When enabling Thinking mode, we recommend setting `do_sample=True` and `temperature=0.6` to mitigate undesired repetition." followed by R1_SYSTEM_PROMPT and `model.system_message = R1_SYSTEMP_PROMPT` (sic - the README assigns a misspelled variable name). R1_SYSTEM_PROMPT text (README.md lines 416-422): "You are an AI assistant that rigorously follows this response protocol: 1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags. 2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline. Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section." To keep it disabled: do not set model.system_message. The default is the conv template's system message (…
- source: README.md lines 80, 410-426; conversation.py lines 384-390; chat_template.jinja; added_tokens.json

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- loader.config_transformers_version: config.json top-level transformers_version is null and llm_config carries none; only generation_config.json says 4.52.1.
- bbox.official_prompt / output_format / coord_range / coord_order: the local README has no grounding prompt or coordinate convention (only a results image under 'Visual Grounding'); README_CN / examples are not shipped in this directory.
- vision_encoder exact checkpoint name: README names the family options (InternViT-300M / InternViT-6B) and the 0.3B vision parameter count for the 8B model, but never states the exact vision checkpoint id; config.json vision_config has no _name_or_path.
- Whether the remote code loads and returns attentions under the transformers versions installed locally (5.x): not tested - no weights were loaded and no code from the model directory was imported.
- Whether the shipped HF-native processor files (InternVLProcessor) produce inputs compatible with the remote-code InternVLChatModel: not documented in the README, not tested.
- PIL resample filter used by image.resize in dynamic_preprocess: the README code passes no resample argument, so it is whatever the installed Pillow defaults to (not stated in the card).

### LLaVA-OneVision-7B

- **hf_repo**: llava-hf/llava-onevision-qwen2-7b-ov-hf
- **revision**: 0d50680527681998e456c7b78950205bedd8a068
- **revision_source**: _logs/hf_download_394198.out line 1. Not listed in MANIFEST.md.
- **size_bytes**: 16073497233
- **license**: apache-2.0
- **license_source**: README.md front matter line 5
- **role**: general
- **base_model**: —

**loader**

- model_class: LlavaOnevisionForConditionalGeneration
- processor_class: LlavaOnevisionProcessor
- trust_remote_code: False
- config_transformers_version: 4.45.0.dev0
- notes: Native transformers class. Card requirement: "transformers >= 4.45.0" (README:50). Card example loads with torch_dtype=torch.float16 (config torch_dtype float16; text_config torch_dtype bfloat16; card "Precision: bfloat16"). Prompt format from chat_template.json: "<|im_start|>role " + all <image> placeholders first, then "\n" + text, "<|im_end|>"; images of a message are rendered before its text regardless of content order. Image token id 151646, video token id 151647. Not covered by the MANIFEST.md smoke test.

**language_model**

- model_type: qwen2
- num_layers: 28
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: text_config does not set use_sliding_window, so Qwen2Config's default False applies; __post_init__ then sets sliding_window=None and every layer_types entry is "full_attention". Grouped-query attention: 28 query heads, 4 key/value heads. RoPE (rope_theta 1e6), 1D positions.
- mlp_type: dense
- head_level_analysis: All 28 layers are standard full causal self-attention with 28 query heads (4 shared kv heads, groups of 7); head-level analysis applies to every layer. Attention-weight return under sdpa/flash backends not checked here.

**vision_encoder**

- type: SigLIP vision transformer (SiglipVisionModel, model_type siglip_vision_model; card says "SO400M"), no pooling head (vision_use_head false)
- num_layers: 26
- hidden_size: 1152
- patch_size: 14
- resolution_strategy: Every view is a fixed 384x384 crop (27x27 = 729 patch tokens, no CLS; conv is "valid" so only 378 of 384 px per side are covered). High resolution is handled by AnyRes tiling, not by a dynamic-resolution ViT: see multi_view. Normalisation mean/std 0.5 (preprocessor_config.json), bicubic resample.
- position_encoding: Learned absolute position embeddings: nn.Embedding(729, 1152) added to patch embeddings (SiglipVisionEmbeddings). Features taken from hidden_states[vision_feature_layer = -1] with vision_feature_select_strategy "full" (nothing dropped).

**connector**

- type: 2-layer MLP (LlavaOnevisionMultiModalProjector): Linear(1152, 3584) -> GELU -> Linear(3584, 3584), plus a learned image_newline vector (3584)
- token_merge: No fixed merge for images (1 patch -> 1 token). Only when the unpadded tile grid exceeds the anyres_max_9 budget (sqrt(h*w / (9*729)) > 1.1) is the tile feature map bilinearly interpolated down to (int(h // ratio), int(w // ratio)). Video frames use a separate 2x pooling path (not relevant to still images).
- tokens_per_image: Single image in a sample (default AnyRes path): N = 729 (base view) + h' * (w' + 1), where (gh*384, gw*384) = select_best_resolution(original (H, W), image_grid_pinpoints), the tile feature map is (27*gh) x (27*gw), unpadding removes the rows or columns that came from aspect-ratio padding giving (h, w), then if ratio = sqrt(h*w / (9*729)) > 1.1: h' = int(h // ratio), w' = int(w // ratio), else h' = h, w' = w; "+1" is one image_newline token appended to every row. The base view gets no newline. Number of vision-tower forward views = 1 + gh*gw. Multi-image sample (more than one image in the same prompt): each image is 729 + 1 = 730 tokens (base view only + one trailing newline).
- token_to_region: Order in the LM sequence for a single image: [729 base tokens] then [tile map rows, each row followed by one newline token]. (1) Base tokens j = 0..728: row j // 27, col j % 27 of the whole image resized directly to 384x384 (aspect ratio NOT preserved on this path), each token a 14x14 px cell of that 384 view, i.e. original x in [col*14*W/384, (col+1)*14*W/384), same for y with H. (2) Tile part, index t = 729 + r*(w'+1) + c: c == w' is the newline; otherwise (r, c) indexes the unpadded high-res feature map. That map comes from the image resized (aspect preserved) into the best resolution (gh*384, gw*384) and centre-padded, cut into gh x gw tiles of 384 in row-major order, each encoded to 27x27, re-assembled (view/permute) into one (27*gh, 27*gw) grid and cropped back to the image area, so (r, c) covers approximately the original-image fraction [c/w, (c+1)/w) x [r/h, (r+1)/h) when no interpolation happened. If the bilinear down-interpolation fired (ratio > 1.1), tokens are blends of neighbouring patches and no longer correspond 1:1 to ViT patches. Multi-image path: each image is padded to a square with the mean colour, resized to 384, tokens are the 27x27 raster of that padded squa…

**multi_view**

- default: Yes for a single image: AnyRes ("anyres_max_9") with 36 grid pinpoints from 384x384 up to 2304x2304 (multiples of 384); the model sees 1 base view (whole image at 384x384) plus gh*gw high-resolution 384x384 tiles, with image_newline tokens and unpadding. When a sample contains more than one image, tiling is switched off automatically by the processor/model (batch_num_images != 1 -> base view only, "only single image patching is supported").
- force_single_view: There is NO officially supported switch for a lone image in transformers 5.5.4. vision_aspect_ratio is only parsed as "anyres_max_N" (the budget for the interpolation step); it does not turn tiling off. Shrinking image_grid_pinpoints to [[384, 384]] (must be set in BOTH the image processor and model.config, since both call select_best_resolution) still yields base + 1 tile = up to 729 + 27*28 tokens, i.e. two views of the image, not one. The only built-in base-only path is the multi-image one (pass a nested list with >= 2 images for the sample; each image becomes 730 tokens). A true single 729(+1)-token view of one image therefore needs a custom override, e.g. calling the model with pixel_values of shape (1, 1, 3, 384, 384), a hand-built prompt with exactly 730 <image> tokens and a batch_num_images value != 1 so that need_patching is False (read from source, NOT executed or verified here), or bypassing the processor and feeding inputs_embeds.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md searched for grounding / bounding box / bbox / point / coordinate - no match.

**thinking**

- has_thinking_mode: False
- how_to_disable: —
- source: chat_template.json:2 has no thinking branch or flag; README.md does not mention thinking/reasoning modes.

**查不到 / 未确认**

- bbox.*: the model card gives no grounding prompt or coordinate convention.
- force_single_view for a lone image: no official switch found in transformers 5.5.4 source; the override sketched above is derived from reading the code and has not been run.
- Whether hidden_states[-1] of the SigLIP tower is taken before or after post_layernorm was not verified by execution (post_layernorm exists at modeling_siglip.py:593,615 and is applied to last_hidden_state).
- Card inconsistency: README:35 says 'LLaVA-Onevision-7b-si was added in August 2024' and config _name_or_path is '/raid/raushan/si-7b', while the repo id is ...-7b-ov-hf; the card says it was copied from the lmms-lab 0.5b-si card (README:21). Which training stage (si vs ov) the weights correspond to cannot be settled from local files beyond the repo name.
- Loading under transformers 5.5.4 not smoke-tested per local records (config written by 4.45.0.dev0).

### MiniCPM-V-4_5

- **hf_repo**: openbmb/MiniCPM-V-4_5
- **revision**: 14a2004908140699317e2109e90573c1e6323c36
- **revision_source**: .cache/huggingface/download/*.metadata 第 1 行（23 个文件一致）。MANIFEST.md 未收录该模型，_logs/MiniCPM-V-4_5_download.log 里没有 repo id / sha。同一字段在 gemma-4-31B-it 上与 MANIFEST 的 sha 一致，据此认定其为 commit sha。
- **size_bytes**: 17408028048
- **license**: apache-2.0
- **license_source**: README.md:15 front matter `license: apache-2.0`；README.md:493 "The MiniCPM-o/V model weights and code are open-sourced under the Apache-2.0 license." 另有 :494 自愿填写的登记问卷（optionally）与 :497-498 免责声明，未见附加的非 Apache 限制条款。
- **role**: general
- **base_model**: —

**loader**

- model_class: MiniCPMV
- processor_class: MiniCPMVProcessor
- trust_remote_code: True
- config_transformers_version: 4.51.0
- notes: README 示例：AutoModel.from_pretrained(..., trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)， 行尾注释 "sdpa or flash_attention_2, no eager"（README.md:263-264）；tokenizer 用 AutoTokenizer(trust_remote_code=True) （tokenizer_class MiniCPMVTokenizerFast）。推理入口是 model.chat(msgs=..., tokenizer=...)（modeling_minicpmv.py:335）。 remote code 直接 `from transformers import Qwen3PreTrainedModel, Qwen3ForCausalLM`（modeling_minicpmv.py:10） 与 `Qwen3Config`（configuration_minicpm.py:8），因此需要自带 Qwen3 的 transformers；本机 5.5.4 有 models/qwen3， 但 config 记录的是 4.51.0，remote code 与 5.5.4 的兼容性未验证。README 未给出版本要求。 chat() 在未传 processor 时用 config._name_or_path 调 AutoProcessor.from_pretrained（:370）； 视觉塔的注意力实现被强制为 eager（除非 flash_attention_2，:36-40，注释 "not suport sdpa"）。 forward 需要显式的 position_ids（:232）。只读目录提示：trust_remote_code 会把模型目录里的 .py 复制到 HF modules cache 再 import，不要把 cwd 设在模型目录内。

**language_model**

- model_type: minicpmv（MiniCPMVConfig 继承 Qwen3Config；LLM 为 transformers 的 Qwen3ForCausalLM(config)，modeling_minicpmv.py:25）
- num_layers: 36
- hidden_size: 4096
- num_attention_heads: 32
- num_key_value_heads: 8
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: config.json：use_sliding_window=false、sliding_window=null、max_window_layers=36，无 layer_types 字段。 按 transformers 5.5.4 的 Qwen3Config（configuration_qwen3.py:87-94）：use_sliding_window 为假时 sliding_window 置 None， layer_types 全部生成为 full_attention；Qwen3Model 只在存在 sliding 层时才建滑窗 mask（modeling_qwen3.py:370, :414-418）。 所以 36 层都是标准因果 softmax 自注意力，GQA：32 个 q 头 / 8 组 KV（每 4 个 q 头共享一组）。rope_theta=1000000。 README.md:26 称 "built on Qwen3-8B and SigLIP2-400M with a total of 8B parameters"。
- mlp_type: dense（Qwen3 MLP，intermediate_size=12288，hidden_act silu）
- head_level_analysis: LLM 的 36 层都可做 head 级分析与注意力屏蔽（全局因果 softmax 注意力）；KV 为 8 组，同组 4 个头共享 K/V。 两点限制：(1) README 示例注明 attn_implementation 只用 sdpa 或 flash_attention_2、"no eager"（README.md:264）， 而取注意力权重通常需要 eager——eager 下能否正常运行未验证； (2) 图像 token 是 resampler 的 learned query 输出，不对应固定图像区域（见 connector.token_to_region）， “按图像区域屏蔽视觉 token”的实验在本模型上只能做到视图/切片粒度。

**vision_encoder**

- type: SiglipVisionTransformer（模型目录自带 modeling_navit_siglip.py，注释 "Copied from HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit and add tgt_sizes"；vision_config.model_type=siglip。README.md:26 称其为 SigLIP2-400M）
- num_layers: 27
- hidden_size: 1152
- patch_size: 14
- resolution_strategy: dynamic（NaViT 式可变分辨率/长宽比，按 tgt_sizes 与 patch_attention_mask 处理不等长 patch 序列）+ 默认切片多视图。 每个视图（整图缩略或一个切片）被缩放到面积约 scale_resolution^2=448x448、长宽比保持、边长为 14 的倍数 （find_best_resize / ensure_divide，image_processing_minicpmv.py:138-154），即每视图约 1024 个 patch。 归一化 mean/std 均为 0.5。vision_config.image_size=980 只决定位置表大小（70x70）。 README.md:36："can process high-resolution images with any aspect ratio and up to 1.8 million pixels (e.g., 1344x1344)"。
- position_encoding: 可学习绝对位置表 nn.Embedding((980/14)^2=4900, hidden)，视作 70x70 网格；对每个视图按 patch 的分数坐标 （arange(0, 1, 1/nb_patches)）bucketize 到 70 个桶，得到 pos_id = bucket_h*70 + bucket_w 后查表相加 （SiglipVisionEmbeddings.forward，modeling_navit_siglip.py:312-353）。

**connector**

- type: Resampler（resampler.py:83-89："A 2D perceiver-resampler network with one cross attention layers by given learnable queries and 2d sincos pos_emb"）。 query_num=64 个可学习 query（nn.Parameter，:112），一层 nn.MultiheadAttention(embed_dim=4096, num_heads=4096//128=32)（:120；modeling_minicpmv.py:50-58）， kv_proj 把 1152 维视觉特征线性投到 4096（:116），K = ln_kv(kv_proj(x)) + 2D sincos 位置编码，V 不加位置（:207-229）， 输出经 ln_post 再乘 proj 矩阵（:239-240）。视频时为“3D resampler”：传入 temporal_ids 后给 K 再加 1D sincos 时间编码， 并把同一组的多帧 patch 拼成一条 KV 序列共同压成 64 个 token（batch_attn_forward，:248-280）。
- token_merge: 是，强压缩：每个视图约 1024 个 patch → 64 个 token（约 16 倍）；不是空间池化，而是 learned-query 交叉注意力重采样。
- tokens_per_image: 64 * (1 + n_slices)。每个视图固定 64 个占位（image_feature_size=64 个 <unk>，get_slice_image_placeholder， image_processing_minicpmv.py:295-313）。切片规则（get_sliced_grid，:264-293）： multiple = min(ceil(W*H / 448^2), max_slice_nums)；multiple <= 1 时不切片（n_slices=0，共 64 token）； 否则在切片数 {multiple-1, multiple, multiple+1}（去掉 1 和超过 max_slice_nums 的）的所有 cols x rows 分解里选 log 长宽比最接近原图的网格，n_slices = cols*rows（2..9）。默认 max_slice_nums=9 → 最多 64*10 = 640 个视觉 token。 序列布局：[<image_id>idx</image_id>]<image>64*unk</image>（整图缩略视图）后接各切片 <slice>64*unk</slice>， 切片按行优先，行与行之间插入 "\n"（get_grid_placeholder，:218-237）。 视觉特征按 image_bound（<image>/<slice> 起止 token 之间的位置）scatter 进 LLM 输入嵌入（processing_minicpmv.py:115-129；modeling_minicpmv.py:175-190）。
- token_to_region: 【不存在固定对应关系】64 个视觉 token 是 64 个 learned query 各自对该视图全部 patch 做一次交叉注意力的输出 （self.attn(self._repeat(q, bs), k, v, key_padding_mask=...)，resampler.py:274-278；query 定义 :112, :210）， 第 j 个 token 不对应网格上的第 j 个位置，也没有行/列顺序；其空间来源由数据相关的注意力权重决定，逐图不同。 能确定的只有视图级对应：第 0 组 64 个 token ↔ 整张图的缩略视图；之后每组 64 个 token ↔ 一个切片， 切片 (row r, col c) 覆盖 refine 后图像的第 r 行第 c 列网格单元（split_to_patches，image_processing_minicpmv.py:178-190； 顺序见 get_sliced_images :257-262）。 可用的近似映射：resampler 交叉注意力图（每视图 64 x (h*w) 的 query→patch 权重，h,w = tgt_sizes）， 可按 patch 行优先（pos_embed[:tgt_h, :tgt_w] reshape，:201）还原成 2D 热图；但现有 forward 只取 self.attn(...)[0]、 丢弃了权重，需要 hook / 重新调用 self.resampler.attn 才能拿到（具体取法未实测）。 此外 patch 特征本身经过 27 层 SigLIP 全局注意力，热图只是近似归因。

**multi_view**

- default: 是。slice_mode=true、max_slice_nums=9（config.json:44,48；preprocessor_config.json）：面积超过 448x448 的图会得到 1 个整图缩略视图 + 2~9 个切片，每个视图 64 token；面积不超过 448^2 的图只有 1 个视图（并被放大到约 448^2 面积）。
- force_single_view: 调用时传 max_slice_nums=1：model.chat(..., max_slice_nums=1)（参数见 modeling_minicpmv.py:348，传给 processor :425） 或 processor(..., max_slice_nums=1)。此时 multiple=min(ceil(ratio),1)=1 → get_sliced_grid 返回 None（:268-270） → slice_image 只生成一个整图视图，按 allow_upscale=True 缩放到约 448^2 面积（:200-205）→ 恒为 64 个视觉 token。 README 的视频示例正是这样用的：`use_image_id=False, max_slice_nums=1`（README.md:419-420）。 不要用 slice_mode=False：那条分支直接返回未缩放的原图（:245-246），且 chat() 断言 config.slice_mode 与 image_processor.slice_mode 相等（modeling_minicpmv.py:377），而 MiniCPMVConfig.__init__ 把 slice_mode 硬写成 True（configuration_minicpm.py:89）。 若要永久生效需同时改 config.slice_config.max_slice_nums 与 image_processor.max_slice_nums（chat() 有相等断言 :376），调用级参数则不受该断言影响。

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md 全文没有 grounding / bounding box 的提示词、输出格式或坐标约定。added_tokens.json 中有 <box>、<ref>、<quad>、<point> 等 token（:13-18），但本地文件未说明用法。

**thinking**

- has_thinking_mode: True
- how_to_disable: model.chat(..., enable_thinking=False)，这是 chat() 的默认值（modeling_minicpmv.py:351）。 README.md:253："If you wish to enable thinking mode, provide the argument `enable_thinking=True` to the chat function."； README.md:270：`enable_thinking=False # If `enable_thinking=True`, the thinking mode is enabled.` chat() 把该参数传给 tokenizer.apply_chat_template（:414）；tokenizer_config.json 的 chat_template 在 generation prompt `<|im_start|>assistant\n` 之后：enable_thinking 为 false 时预填 `<think>\n\n</think>\n\n`，为 true 时预填 `<think>\n`（模板第 80-87 行）； 若自己调模板且不传 enable_thinking，则两者都不预填。generation_config.json:12 也写了 chat_template_kwargs.enable_thinking=false。 开启思考时 chat() 的采样参数不同（关闭时额外加 top_p=0.8, top_k=100, repetition_penalty=1.03，:437-444）。
- source: README.md:33,53,253,270；modeling_minicpmv.py:351,414-420；tokenizer_config.json chat_template；generation_config.json:12

**查不到 / 未确认**

- connector.token_to_region：不存在固定规则（resampler learned query）。交叉注意力图作为近似映射的具体取法（hook nn.MultiheadAttention 取权重）未实测。
- bbox 全部字段：README 没有 grounding 提示词/格式/坐标约定；<box>/<ref>/<quad>/<point> token 存在但用法未在本地文件中说明。
- transformers 版本要求：README 未写；config 记录 4.51.0；remote code 依赖 transformers 内置 Qwen3 类，与本机 5.5.4 的兼容性未加载验证。
- attn_implementation='eager' 是否可用：README 注释写 "no eager" 但未说明原因，未实测；这直接影响能否取 LLM 注意力权重。
- revision 未出现在 MANIFEST.md 或下载日志中，只能取自 .cache 的下载元数据。
- vision encoder 是 SigLIP 还是 SigLIP2：README 说 SigLIP2-400M，config/代码标的是 siglip（so400m-14-980 navit 改写），以本地代码结构为准，权重来源无法从本地文件确认。
- 每视图 patch 数（约 1024）与 16 倍压缩率是按 find_best_resize 公式推算，未用真实图像验证。

### Qwen3.5-2B

- **hf_repo**: Qwen/Qwen3.5-2B
- **revision**: 15852e8c16360a2fea060d615a32b45270f8a8fc
- **revision_source**: _logs/hf_download_394169.out line 1 (download line "Qwen/Qwen3.5-2B@<sha>"); this model is NOT listed in MANIFEST.md
- **size_bytes**: 4571274023
- **license**: apache-2.0
- **license_source**: README.md:3 front matter (license: apache-2.0); LICENSE:1-2 (Apache License, Version 2.0)
- **role**: general
- **base_model**: —

**loader**

- model_class: Qwen3_5ForConditionalGeneration
- processor_class: Qwen3VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.57.0.dev0
- notes: ['model_type=qwen3_5; config.json architectures=[Qwen3_5ForConditionalGeneration], which is also the AutoModelForImageTextToText mapping in transformers 5.5.4 (models/auto/modeling_auto.py:1016).', 'preprocessor_config.json names processor_class=Qwen3VLProcessor and image_processor_type=Qwen2VLImageProcessorFast; in transformers 5.5.4 the image-processor mapping for qwen3_5 is Qwen2VLImageProcessor (torchvision backend) with the legacy "Fast" name resolved to it (models/auto/image_processing_auto.py:218). Tokenizer mapping: Qwen3_5Tokenizer; video: Qwen3VLVideoProcessor.', 'trust_remote_code not needed: the model directory contains no .py files and tokenizer_config.json has no auto_map; all classes ship in the installed transformers 5.5.4 (spatialcraft env).', 'config.json was written by transformers 4.57.0.dev0; the code read for this file is 5.5.4.', 'Gated DeltaNet fast path needs the optional packages flash-linear-attention (fla) and causal-conv1d; without them the pure-torch implementation is used (modeling_qwen3_5.py:205-207). pip list of the spatialcraft env shows neither package.', 'README:678 says the latest transformers is required and documents only serving (transformer…

**language_model**

- model_type: qwen3_5_text
- num_layers: 24
- hidden_size: 2048
- num_attention_heads: 8
- num_key_value_heads: 2
- head_dim: 256
- layer_mixers（逐层，0 起）: `L L L F L L L F L L L F L L L F L L L F L L L F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: linear_attention layers (indices i with (i+1)%4 != 0; 18 of 24 layers) are instances of Qwen3_5GatedDeltaNet (modeling_qwen3_5.py:356), built at modeling_qwen3_5.py:736-737. It is a Gated DeltaNet recurrent/linear-time mixer, not softmax attention: the README calls the block "Gated DeltaNet" (README:52). Mechanism from source: in_proj_qkv projects the hidden state to q,k,v; a depthwise causal Conv1d with kernel 4 (linear_conv_kernel_dim) + SiLU is applied to the concatenated qkv channels (modeling_qwen3_5.py:375-382); q and k are L2-normalised; a per-head matrix state S (key_dim x value_dim per head) is updated per position with the gated delta rule S_t = exp(g_t)*S_(t-1) + k_t (x) beta_t*(v_t - S k_t), output o_t = S_t q_t (reference loop modeling_qwen3_5.py:314-353; chunked parallel form used for full sequences modeling_qwen3_5.py:234, called at modeling_qwen3_5.py:499); beta = sigmoid(in_proj_b), g = -exp(A_log)*softplus(in_proj_a + dt_bias). The output is passed through a gated RMSNorm (norm, then multiply by SiLU(z) with z = in_proj_z(hidden)) and out_proj (modeling_qwen3_5.py:529-532). Heads: linear_num_key_heads=16 (q/k head dim 128), linear_num_value_heads=16 (value head d…
- mlp_type: dense: Qwen3_5MLP, SwiGLU-style gate_proj/up_proj/down_proj with SiLU, intermediate_size=6144 (modeling_qwen3_5.py:695-708, built at 740). No expert/MoE keys in config.json; README:52 lists FFN.
- head_level_analysis: Layers that admit standard head-level analysis: the 6 full_attention layers, indices [3, 7, 11, 15, 19, 23] (every 4th layer, (i+1)%4==0). Why: each has 8 softmax attention heads (Qwen3_5Attention); the per-head outputs are still separate (shape [batch, seq, 8*256], head h = slice h*256:(h+1)*256) at the input of self_attn.o_proj (modeling_qwen3_5.py:691), so a forward-pre-hook on o_proj reads or ablates single heads (that tensor is already multiplied by the sigmoid output gate, modeling_qwen3_5.py:689). Position-to-position attention masking is possible because these layers take a 4D causal mask built by create_causal_mask (modeling_qwen3_5.py:1260, selected at 1273) that is added to the logits before softmax in the eager path (modeling_qwen3_5.py:607-609). Attention weights require attn_implementation="eager"; under the default sdpa they are None. With eager, output_attentions yields one entry per full_attention layer only (hooks are installed on Qwen3_5Attention modules and None outputs are skipped: modeling_qwen3_5.py:795-799, utils/output_capturing.py:103-117), so the returned tuple has 6 entries and entry j corresponds to layer index 4*j+3, not to layer j. GQA: 2 KV heads, s…

**vision_encoder**

- type: Qwen3_5VisionModel (modeling_qwen3_5.py:995): ViT with a Conv3d patch embedding (kernel/stride = temporal 2 x 16 x 16, modeling_qwen3_5.py:828-837), 24 pre-norm blocks with 16-head non-causal attention (modeling_qwen3_5.py:878), MLP width 4096, output projected to 2048 by the patch merger. The README only says "Vision Encoder" (README:45) and does not name the pretrained encoder family.
- num_layers: 24
- hidden_size: 1024
- patch_size: 16
- resolution_strategy: dynamic, single view. Image processor resizes each image with smart_resize(factor=32=patch 16 x merge 2): both sides rounded to multiples of 32, aspect ratio kept, total pixels clamped to [min_pixels, max_pixels] = [size.shortest_edge, size.longest_edge] = [65536, 16777216] (preprocessor_config.json; image_processing_qwen2_vl.py:62-88, 171-177). Aspect ratio above 200 raises. Resample is the class default BICUBIC (image_processing_qwen2_vl.py:94); mean/std 0.5. Default range is therefore 64 to 16384 LM tokens per image.
- position_encoding: Two schemes together. (1) Learned absolute table pos_embed = nn.Embedding(2304, 1024) treated as a 48x48 grid and bilinearly interpolated to the actual (grid_h, grid_w) of each image, then added to the patch embeddings (modeling_qwen3_5.py:1014-1015, 1070-1131). (2) 2D rotary embedding over (row, col) patch coordinates applied to q/k in every vision attention block (modeling_qwen3_5.py:1030-1068).

**connector**

- type: Qwen3_5VisionPatchMerger (modeling_qwen3_5.py:848-861): LayerNorm on each patch, concatenate the 4 patches of each 2x2 block, then Linear -> GELU -> Linear to out_hidden_size=2048. deepstack_visual_indexes is [] in config.json and the transformers 5.5.4 Qwen3.5 code has no deepstack path (removed in modular_qwen3_5.py), so visual features enter the LM only once, by replacing the <|image_pad|> input embeddings (modeling_qwen3_5.py:1648).
- token_merge: yes: spatial_merge_size=2, 2x2 patches -> 1 token (4x fewer tokens than patches); 32x32 resized pixels per LM token.
- tokens_per_image: N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216) is the resized size (multiples of 32). Equivalently image_grid_thw.prod()/4 with grid = (1, H2/16, W2/16) (processing_qwen3_vl.py:128-133; modeling_qwen3_5.py:1515). The chat template wraps them as <|vision_start|> + N x <|image_pad|> + <|vision_end|> (chat_template.jinja:18), i.e. N+2 sequence positions per image.
- token_to_region: Raster (row-major) order at the LM-token level. Let mw = W2/32, mh = H2/32. The k-th <|image_pad|> position of an image (k from 0) is merged cell row r = k // mw, col c = k % mw, covering resized-image pixels x in [32c, 32c+32), y in [32r, 32r+32); multiply by W/W2 and H/H2 to get original-image pixels. Reason: the image processor emits patches block-major, in order (block_row, block_col, intra_row, intra_col) (image_processing_qwen2_vl.py:203-220), the pos-embed is permuted the same way (modeling_qwen3_5.py:1125-1127), and the merger concatenates every 4 consecutive patches (modeling_qwen3_5.py:859), which are exactly one 2x2 block. Consequence for hooks INSIDE the vision encoder: patch index p is not raster; block b = p // 4, j = p % 4, patch row = 2*(b // mw) + j // 2, patch col = 2*(b % mw) + j % 2. A still image is duplicated along time to fill temporal_patch_size=2, giving grid_t = 1 (image_processing_qwen2_vl.py:196-200).

**multi_view**

- default: single view: no tiling, slicing or thumbnail. The image processor only resizes and patchifies each image into one grid (image_processing_qwen2_vl.py:166-232) and the processor inserts one placeholder run per image (processing_qwen3_vl.py:127-135).
- force_single_view: Nothing to turn off. To fix the token count, either pre-resize the image yourself to an exact (H2, W2) with both sides multiples of 32 and 65536 <= H2*W2 <= 16777216 (smart_resize then leaves it unchanged, giving exactly H2*W2/1024 tokens), or narrow the range by passing min_pixels and max_pixels together (both are needed for a per-call override, image_processing_qwen2_vl.py:132-133) or size={"shortest_edge": ..., "longest_edge": ...} to the processor / image processor. With min_pixels == max_pixels the count is only approximately fixed because of rounding to multiples of 32. Not run-tested here.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated locally. README.md was searched for grounding / bounding box / bbox / bbox_2d / point_2d / coordinate / normalized / 0-1000: the only hits are benchmark table rows (RefCOCO / ODInW) at README lines [429, 443], with no prompt or coordinate convention. chat_template.jinja has no grounding instructions.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is OFF by default for this model. chat_template.jinja:149-153 opens a reasoning block ('<think>\n') only when the template variable enable_thinking is defined and true; otherwise it emits an empty block '<think>\n\n</think>\n\n' after '<|im_start|>assistant\n'. So: do not pass enable_thinking (or pass enable_thinking=False). README:585 confirms: "Qwen3.5-2B operates in non-thinking mode by default". README shows enabling via API extra_body/chat_template_kwargs {"enable_thinking": True}. README:826: the /think and /nothink soft switches are not officially supported.
- source: chat_template.jinja:149-153; README.md:585, 826

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- bbox.official_prompt / output_format / coord_range / coord_order: the local README.md contains no grounding or bounding-box prompt and no coordinate convention (only benchmark table rows at README lines [429, 443]); chat_template.jinja and the processor source contain none either. Not imported from other Qwen generations.
- vision_encoder pretrained origin / family name: not stated in README or config; only the architecture is known from source.
- text_config keys that the transformers 5.5.4 modeling code never reads (grep of the modeling file), so their intended effect is unknown from local files: attn_output_gate, mlp_only_layers, mtp_num_hidden_layers, mtp_use_dedicated_embeddings, mamba_ssm_dtype, full_attention_interval (attn_output_gate: the gate is hard-coded on; full_attention_interval: only used when layer_types is absent; mtp weights are ignored on load).
- All head_level_analysis / output_attentions statements come from reading source; no forward pass was run (login node, no weights loaded).
- In-process loading example (AutoProcessor/AutoModel snippet) and passing enable_thinking via apply_chat_template: README shows only the serving API form.
- size_bytes = sum of local file sizes excluding .cache (13 files, computed with find/stat); not recorded in MANIFEST.md or _logs. License string comes from README front matter and LICENSE only (no MANIFEST entry).

### Qwen3.5-4B

- **hf_repo**: Qwen/Qwen3.5-4B
- **revision**: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
- **revision_source**: MANIFEST.md line 16
- **size_bytes**: 9342907469
- **license**: apache-2.0
- **license_source**: README.md:3 front matter (license: apache-2.0); LICENSE:1-2 (Apache License, Version 2.0); MANIFEST.md line 16
- **role**: general
- **base_model**: —

**loader**

- model_class: Qwen3_5ForConditionalGeneration
- processor_class: Qwen3VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.57.0.dev0
- notes: ['model_type=qwen3_5; config.json architectures=[Qwen3_5ForConditionalGeneration], which is also the AutoModelForImageTextToText mapping in transformers 5.5.4 (models/auto/modeling_auto.py:1016).', 'preprocessor_config.json names processor_class=Qwen3VLProcessor and image_processor_type=Qwen2VLImageProcessorFast; in transformers 5.5.4 the image-processor mapping for qwen3_5 is Qwen2VLImageProcessor (torchvision backend) with the legacy "Fast" name resolved to it (models/auto/image_processing_auto.py:218). Tokenizer mapping: Qwen3_5Tokenizer; video: Qwen3VLVideoProcessor.', 'trust_remote_code not needed: the model directory contains no .py files and tokenizer_config.json has no auto_map; all classes ship in the installed transformers 5.5.4 (spatialcraft env).', 'config.json was written by transformers 4.57.0.dev0; the code read for this file is 5.5.4.', 'Gated DeltaNet fast path needs the optional packages flash-linear-attention (fla) and causal-conv1d; without them the pure-torch implementation is used (modeling_qwen3_5.py:205-207). pip list of the spatialcraft env shows neither package.', 'README:808 says the latest transformers is required and documents only serving (transformer…

**language_model**

- model_type: qwen3_5_text
- num_layers: 32
- hidden_size: 2560
- num_attention_heads: 16
- num_key_value_heads: 4
- head_dim: 256
- layer_mixers（逐层，0 起）: `L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: linear_attention layers (indices i with (i+1)%4 != 0; 24 of 32 layers) are instances of Qwen3_5GatedDeltaNet (modeling_qwen3_5.py:356), built at modeling_qwen3_5.py:736-737. It is a Gated DeltaNet recurrent/linear-time mixer, not softmax attention: the README calls the block "Gated DeltaNet" (README:52). Mechanism from source: in_proj_qkv projects the hidden state to q,k,v; a depthwise causal Conv1d with kernel 4 (linear_conv_kernel_dim) + SiLU is applied to the concatenated qkv channels (modeling_qwen3_5.py:375-382); q and k are L2-normalised; a per-head matrix state S (key_dim x value_dim per head) is updated per position with the gated delta rule S_t = exp(g_t)*S_(t-1) + k_t (x) beta_t*(v_t - S k_t), output o_t = S_t q_t (reference loop modeling_qwen3_5.py:314-353; chunked parallel form used for full sequences modeling_qwen3_5.py:234, called at modeling_qwen3_5.py:499); beta = sigmoid(in_proj_b), g = -exp(A_log)*softplus(in_proj_a + dt_bias). The output is passed through a gated RMSNorm (norm, then multiply by SiLU(z) with z = in_proj_z(hidden)) and out_proj (modeling_qwen3_5.py:529-532). Heads: linear_num_key_heads=16 (q/k head dim 128), linear_num_value_heads=32 (value head d…
- mlp_type: dense: Qwen3_5MLP, SwiGLU-style gate_proj/up_proj/down_proj with SiLU, intermediate_size=9216 (modeling_qwen3_5.py:695-708, built at 740). No expert/MoE keys in config.json; README:52 lists FFN.
- head_level_analysis: Layers that admit standard head-level analysis: the 8 full_attention layers, indices [3, 7, 11, 15, 19, 23, 27, 31] (every 4th layer, (i+1)%4==0). Why: each has 16 softmax attention heads (Qwen3_5Attention); the per-head outputs are still separate (shape [batch, seq, 16*256], head h = slice h*256:(h+1)*256) at the input of self_attn.o_proj (modeling_qwen3_5.py:691), so a forward-pre-hook on o_proj reads or ablates single heads (that tensor is already multiplied by the sigmoid output gate, modeling_qwen3_5.py:689). Position-to-position attention masking is possible because these layers take a 4D causal mask built by create_causal_mask (modeling_qwen3_5.py:1260, selected at 1273) that is added to the logits before softmax in the eager path (modeling_qwen3_5.py:607-609). Attention weights require attn_implementation="eager"; under the default sdpa they are None. With eager, output_attentions yields one entry per full_attention layer only (hooks are installed on Qwen3_5Attention modules and None outputs are skipped: modeling_qwen3_5.py:795-799, utils/output_capturing.py:103-117), so the returned tuple has 8 entries and entry j corresponds to layer index 4*j+3, not to layer j. GQA: 4 K…

**vision_encoder**

- type: Qwen3_5VisionModel (modeling_qwen3_5.py:995): ViT with a Conv3d patch embedding (kernel/stride = temporal 2 x 16 x 16, modeling_qwen3_5.py:828-837), 24 pre-norm blocks with 16-head non-causal attention (modeling_qwen3_5.py:878), MLP width 4096, output projected to 2560 by the patch merger. The README only says "Vision Encoder" (README:45) and does not name the pretrained encoder family.
- num_layers: 24
- hidden_size: 1024
- patch_size: 16
- resolution_strategy: dynamic, single view. Image processor resizes each image with smart_resize(factor=32=patch 16 x merge 2): both sides rounded to multiples of 32, aspect ratio kept, total pixels clamped to [min_pixels, max_pixels] = [size.shortest_edge, size.longest_edge] = [65536, 16777216] (preprocessor_config.json; image_processing_qwen2_vl.py:62-88, 171-177). Aspect ratio above 200 raises. Resample is the class default BICUBIC (image_processing_qwen2_vl.py:94); mean/std 0.5. Default range is therefore 64 to 16384 LM tokens per image.
- position_encoding: Two schemes together. (1) Learned absolute table pos_embed = nn.Embedding(2304, 1024) treated as a 48x48 grid and bilinearly interpolated to the actual (grid_h, grid_w) of each image, then added to the patch embeddings (modeling_qwen3_5.py:1014-1015, 1070-1131). (2) 2D rotary embedding over (row, col) patch coordinates applied to q/k in every vision attention block (modeling_qwen3_5.py:1030-1068).

**connector**

- type: Qwen3_5VisionPatchMerger (modeling_qwen3_5.py:848-861): LayerNorm on each patch, concatenate the 4 patches of each 2x2 block, then Linear -> GELU -> Linear to out_hidden_size=2560. deepstack_visual_indexes is [] in config.json and the transformers 5.5.4 Qwen3.5 code has no deepstack path (removed in modular_qwen3_5.py), so visual features enter the LM only once, by replacing the <|image_pad|> input embeddings (modeling_qwen3_5.py:1648).
- token_merge: yes: spatial_merge_size=2, 2x2 patches -> 1 token (4x fewer tokens than patches); 32x32 resized pixels per LM token.
- tokens_per_image: N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216) is the resized size (multiples of 32). Equivalently image_grid_thw.prod()/4 with grid = (1, H2/16, W2/16) (processing_qwen3_vl.py:128-133; modeling_qwen3_5.py:1515). The chat template wraps them as <|vision_start|> + N x <|image_pad|> + <|vision_end|> (chat_template.jinja:18), i.e. N+2 sequence positions per image.
- token_to_region: Raster (row-major) order at the LM-token level. Let mw = W2/32, mh = H2/32. The k-th <|image_pad|> position of an image (k from 0) is merged cell row r = k // mw, col c = k % mw, covering resized-image pixels x in [32c, 32c+32), y in [32r, 32r+32); multiply by W/W2 and H/H2 to get original-image pixels. Reason: the image processor emits patches block-major, in order (block_row, block_col, intra_row, intra_col) (image_processing_qwen2_vl.py:203-220), the pos-embed is permuted the same way (modeling_qwen3_5.py:1125-1127), and the merger concatenates every 4 consecutive patches (modeling_qwen3_5.py:859), which are exactly one 2x2 block. Consequence for hooks INSIDE the vision encoder: patch index p is not raster; block b = p // 4, j = p % 4, patch row = 2*(b // mw) + j // 2, patch col = 2*(b % mw) + j % 2. A still image is duplicated along time to fill temporal_patch_size=2, giving grid_t = 1 (image_processing_qwen2_vl.py:196-200).

**multi_view**

- default: single view: no tiling, slicing or thumbnail. The image processor only resizes and patchifies each image into one grid (image_processing_qwen2_vl.py:166-232) and the processor inserts one placeholder run per image (processing_qwen3_vl.py:127-135).
- force_single_view: Nothing to turn off. To fix the token count, either pre-resize the image yourself to an exact (H2, W2) with both sides multiples of 32 and 65536 <= H2*W2 <= 16777216 (smart_resize then leaves it unchanged, giving exactly H2*W2/1024 tokens), or narrow the range by passing min_pixels and max_pixels together (both are needed for a per-call override, image_processing_qwen2_vl.py:132-133) or size={"shortest_edge": ..., "longest_edge": ...} to the processor / image processor. With min_pixels == max_pixels the count is only approximately fixed because of rounding to multiples of 32. Not run-tested here.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated locally. README.md was searched for grounding / bounding box / bbox / bbox_2d / point_2d / coordinate / normalized / 0-1000: the only hits are benchmark table rows (RefCOCO / ODInW) at README lines [524], with no prompt or coordinate convention. chat_template.jinja has no grounding instructions.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is ON by default. chat_template.jinja:149-153: when the template variable enable_thinking is defined and false the generation prompt ends with an empty block '<think>\n\n</think>\n\n'; otherwise it ends with an open '<think>\n'. To disable, set enable_thinking=False as a chat-template variable. README:959 states the model thinks by default; README:994 shows the serving form "chat_template_kwargs": {"enable_thinking": False}. README:957: the /think and /nothink soft switches are not officially supported.
- source: chat_template.jinja:149-153; README.md:959, 994, 957

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- bbox.official_prompt / output_format / coord_range / coord_order: the local README.md contains no grounding or bounding-box prompt and no coordinate convention (only benchmark table rows at README lines [524]); chat_template.jinja and the processor source contain none either. Not imported from other Qwen generations.
- vision_encoder pretrained origin / family name: not stated in README or config; only the architecture is known from source.
- text_config keys that the transformers 5.5.4 modeling code never reads (grep of the modeling file), so their intended effect is unknown from local files: attn_output_gate, mlp_only_layers, mtp_num_hidden_layers, mtp_use_dedicated_embeddings, mamba_ssm_dtype, full_attention_interval (attn_output_gate: the gate is hard-coded on; full_attention_interval: only used when layer_types is absent; mtp weights are ignored on load).
- All head_level_analysis / output_attentions statements come from reading source; no forward pass was run (login node, no weights loaded).
- In-process loading example (AutoProcessor/AutoModel snippet) and passing enable_thinking via apply_chat_template: README shows only the serving API form.

### Qwen3.5-9B

- **hf_repo**: Qwen/Qwen3.5-9B
- **revision**: c202236235762e1c871ad0ccb60c8ee5ba337b9a
- **revision_source**: MANIFEST.md line 9
- **size_bytes**: 19329393661
- **license**: apache-2.0
- **license_source**: README.md:3 front matter (license: apache-2.0); LICENSE:1-2 (Apache License, Version 2.0); MANIFEST.md line 9
- **role**: general
- **base_model**: —

**loader**

- model_class: Qwen3_5ForConditionalGeneration
- processor_class: Qwen3VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.57.0.dev0
- notes: ['model_type=qwen3_5; config.json architectures=[Qwen3_5ForConditionalGeneration], which is also the AutoModelForImageTextToText mapping in transformers 5.5.4 (models/auto/modeling_auto.py:1016).', 'preprocessor_config.json names processor_class=Qwen3VLProcessor and image_processor_type=Qwen2VLImageProcessorFast; in transformers 5.5.4 the image-processor mapping for qwen3_5 is Qwen2VLImageProcessor (torchvision backend) with the legacy "Fast" name resolved to it (models/auto/image_processing_auto.py:218). Tokenizer mapping: Qwen3_5Tokenizer; video: Qwen3VLVideoProcessor.', 'trust_remote_code not needed: the model directory contains no .py files and tokenizer_config.json has no auto_map; all classes ship in the installed transformers 5.5.4 (spatialcraft env).', 'config.json was written by transformers 4.57.0.dev0; the code read for this file is 5.5.4.', 'Gated DeltaNet fast path needs the optional packages flash-linear-attention (fla) and causal-conv1d; without them the pure-torch implementation is used (modeling_qwen3_5.py:205-207). pip list of the spatialcraft env shows neither package.', 'README:806 says the latest transformers is required and documents only serving (transformer…

**language_model**

- model_type: qwen3_5_text
- num_layers: 32
- hidden_size: 4096
- num_attention_heads: 16
- num_key_value_heads: 4
- head_dim: 256
- layer_mixers（逐层，0 起）: `L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: linear_attention layers (indices i with (i+1)%4 != 0; 24 of 32 layers) are instances of Qwen3_5GatedDeltaNet (modeling_qwen3_5.py:356), built at modeling_qwen3_5.py:736-737. It is a Gated DeltaNet recurrent/linear-time mixer, not softmax attention: the README calls the block "Gated DeltaNet" (README:52). Mechanism from source: in_proj_qkv projects the hidden state to q,k,v; a depthwise causal Conv1d with kernel 4 (linear_conv_kernel_dim) + SiLU is applied to the concatenated qkv channels (modeling_qwen3_5.py:375-382); q and k are L2-normalised; a per-head matrix state S (key_dim x value_dim per head) is updated per position with the gated delta rule S_t = exp(g_t)*S_(t-1) + k_t (x) beta_t*(v_t - S k_t), output o_t = S_t q_t (reference loop modeling_qwen3_5.py:314-353; chunked parallel form used for full sequences modeling_qwen3_5.py:234, called at modeling_qwen3_5.py:499); beta = sigmoid(in_proj_b), g = -exp(A_log)*softplus(in_proj_a + dt_bias). The output is passed through a gated RMSNorm (norm, then multiply by SiLU(z) with z = in_proj_z(hidden)) and out_proj (modeling_qwen3_5.py:529-532). Heads: linear_num_key_heads=16 (q/k head dim 128), linear_num_value_heads=32 (value head d…
- mlp_type: dense: Qwen3_5MLP, SwiGLU-style gate_proj/up_proj/down_proj with SiLU, intermediate_size=12288 (modeling_qwen3_5.py:695-708, built at 740). No expert/MoE keys in config.json; README:52 lists FFN.
- head_level_analysis: Layers that admit standard head-level analysis: the 8 full_attention layers, indices [3, 7, 11, 15, 19, 23, 27, 31] (every 4th layer, (i+1)%4==0). Why: each has 16 softmax attention heads (Qwen3_5Attention); the per-head outputs are still separate (shape [batch, seq, 16*256], head h = slice h*256:(h+1)*256) at the input of self_attn.o_proj (modeling_qwen3_5.py:691), so a forward-pre-hook on o_proj reads or ablates single heads (that tensor is already multiplied by the sigmoid output gate, modeling_qwen3_5.py:689). Position-to-position attention masking is possible because these layers take a 4D causal mask built by create_causal_mask (modeling_qwen3_5.py:1260, selected at 1273) that is added to the logits before softmax in the eager path (modeling_qwen3_5.py:607-609). Attention weights require attn_implementation="eager"; under the default sdpa they are None. With eager, output_attentions yields one entry per full_attention layer only (hooks are installed on Qwen3_5Attention modules and None outputs are skipped: modeling_qwen3_5.py:795-799, utils/output_capturing.py:103-117), so the returned tuple has 8 entries and entry j corresponds to layer index 4*j+3, not to layer j. GQA: 4 K…

**vision_encoder**

- type: Qwen3_5VisionModel (modeling_qwen3_5.py:995): ViT with a Conv3d patch embedding (kernel/stride = temporal 2 x 16 x 16, modeling_qwen3_5.py:828-837), 27 pre-norm blocks with 16-head non-causal attention (modeling_qwen3_5.py:878), MLP width 4304, output projected to 4096 by the patch merger. The README only says "Vision Encoder" (README:45) and does not name the pretrained encoder family.
- num_layers: 27
- hidden_size: 1152
- patch_size: 16
- resolution_strategy: dynamic, single view. Image processor resizes each image with smart_resize(factor=32=patch 16 x merge 2): both sides rounded to multiples of 32, aspect ratio kept, total pixels clamped to [min_pixels, max_pixels] = [size.shortest_edge, size.longest_edge] = [65536, 16777216] (preprocessor_config.json; image_processing_qwen2_vl.py:62-88, 171-177). Aspect ratio above 200 raises. Resample is the class default BICUBIC (image_processing_qwen2_vl.py:94); mean/std 0.5. Default range is therefore 64 to 16384 LM tokens per image.
- position_encoding: Two schemes together. (1) Learned absolute table pos_embed = nn.Embedding(2304, 1152) treated as a 48x48 grid and bilinearly interpolated to the actual (grid_h, grid_w) of each image, then added to the patch embeddings (modeling_qwen3_5.py:1014-1015, 1070-1131). (2) 2D rotary embedding over (row, col) patch coordinates applied to q/k in every vision attention block (modeling_qwen3_5.py:1030-1068).

**connector**

- type: Qwen3_5VisionPatchMerger (modeling_qwen3_5.py:848-861): LayerNorm on each patch, concatenate the 4 patches of each 2x2 block, then Linear -> GELU -> Linear to out_hidden_size=4096. deepstack_visual_indexes is [] in config.json and the transformers 5.5.4 Qwen3.5 code has no deepstack path (removed in modular_qwen3_5.py), so visual features enter the LM only once, by replacing the <|image_pad|> input embeddings (modeling_qwen3_5.py:1648).
- token_merge: yes: spatial_merge_size=2, 2x2 patches -> 1 token (4x fewer tokens than patches); 32x32 resized pixels per LM token.
- tokens_per_image: N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216) is the resized size (multiples of 32). Equivalently image_grid_thw.prod()/4 with grid = (1, H2/16, W2/16) (processing_qwen3_vl.py:128-133; modeling_qwen3_5.py:1515). The chat template wraps them as <|vision_start|> + N x <|image_pad|> + <|vision_end|> (chat_template.jinja:18), i.e. N+2 sequence positions per image.
- token_to_region: Raster (row-major) order at the LM-token level. Let mw = W2/32, mh = H2/32. The k-th <|image_pad|> position of an image (k from 0) is merged cell row r = k // mw, col c = k % mw, covering resized-image pixels x in [32c, 32c+32), y in [32r, 32r+32); multiply by W/W2 and H/H2 to get original-image pixels. Reason: the image processor emits patches block-major, in order (block_row, block_col, intra_row, intra_col) (image_processing_qwen2_vl.py:203-220), the pos-embed is permuted the same way (modeling_qwen3_5.py:1125-1127), and the merger concatenates every 4 consecutive patches (modeling_qwen3_5.py:859), which are exactly one 2x2 block. Consequence for hooks INSIDE the vision encoder: patch index p is not raster; block b = p // 4, j = p % 4, patch row = 2*(b // mw) + j // 2, patch col = 2*(b % mw) + j % 2. A still image is duplicated along time to fill temporal_patch_size=2, giving grid_t = 1 (image_processing_qwen2_vl.py:196-200).

**multi_view**

- default: single view: no tiling, slicing or thumbnail. The image processor only resizes and patchifies each image into one grid (image_processing_qwen2_vl.py:166-232) and the processor inserts one placeholder run per image (processing_qwen3_vl.py:127-135).
- force_single_view: Nothing to turn off. To fix the token count, either pre-resize the image yourself to an exact (H2, W2) with both sides multiples of 32 and 65536 <= H2*W2 <= 16777216 (smart_resize then leaves it unchanged, giving exactly H2*W2/1024 tokens), or narrow the range by passing min_pixels and max_pixels together (both are needed for a per-call override, image_processing_qwen2_vl.py:132-133) or size={"shortest_edge": ..., "longest_edge": ...} to the processor / image processor. With min_pixels == max_pixels the count is only approximately fixed because of rounding to multiples of 32. Not run-tested here.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated locally. README.md was searched for grounding / bounding box / bbox / bbox_2d / point_2d / coordinate / normalized / 0-1000: the only hits are benchmark table rows (RefCOCO / ODInW) at README lines [524], with no prompt or coordinate convention. chat_template.jinja has no grounding instructions.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is ON by default. chat_template.jinja:149-153: when the template variable enable_thinking is defined and false the generation prompt ends with an empty block '<think>\n\n</think>\n\n'; otherwise it ends with an open '<think>\n'. To disable, set enable_thinking=False as a chat-template variable. README:957 states the model thinks by default; README:992 shows the serving form "chat_template_kwargs": {"enable_thinking": False}. README:955: the /think and /nothink soft switches are not officially supported.
- source: chat_template.jinja:149-153; README.md:957, 992, 955

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- bbox.official_prompt / output_format / coord_range / coord_order: the local README.md contains no grounding or bounding-box prompt and no coordinate convention (only benchmark table rows at README lines [524]); chat_template.jinja and the processor source contain none either. Not imported from other Qwen generations.
- vision_encoder pretrained origin / family name: not stated in README or config; only the architecture is known from source.
- text_config keys that the transformers 5.5.4 modeling code never reads (grep of the modeling file), so their intended effect is unknown from local files: attn_output_gate, mlp_only_layers, mtp_num_hidden_layers, mtp_use_dedicated_embeddings, mamba_ssm_dtype, full_attention_interval (attn_output_gate: the gate is hard-coded on; full_attention_interval: only used when layer_types is absent; mtp weights are ignored on load).
- All head_level_analysis / output_attentions statements come from reading source; no forward pass was run (login node, no weights loaded).
- In-process loading example (AutoProcessor/AutoModel snippet) and passing enable_thinking via apply_chat_template: README shows only the serving API form.

### Qwen3.6-27B

- **hf_repo**: Qwen/Qwen3.6-27B
- **revision**: 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
- **revision_source**: MANIFEST.md line 10
- **size_bytes**: 55586107940
- **license**: apache-2.0
- **license_source**: README.md:3 front matter (license: apache-2.0); LICENSE:1-2 (Apache License, Version 2.0); MANIFEST.md line 10
- **role**: general
- **base_model**: —

**loader**

- model_class: Qwen3_5ForConditionalGeneration
- processor_class: Qwen3VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.57.1
- notes: ['model_type=qwen3_5; config.json architectures=[Qwen3_5ForConditionalGeneration], which is also the AutoModelForImageTextToText mapping in transformers 5.5.4 (models/auto/modeling_auto.py:1016).', 'preprocessor_config.json names processor_class=Qwen3VLProcessor and image_processor_type=Qwen2VLImageProcessorFast; in transformers 5.5.4 the image-processor mapping for qwen3_5 is Qwen2VLImageProcessor (torchvision backend) with the legacy "Fast" name resolved to it (models/auto/image_processing_auto.py:218). Tokenizer mapping: Qwen3_5Tokenizer; video: Qwen3VLVideoProcessor.', 'trust_remote_code not needed: the model directory contains no .py files and tokenizer_config.json has no auto_map; all classes ship in the installed transformers 5.5.4 (spatialcraft env).', 'config.json was written by transformers 4.57.1; the code read for this file is 5.5.4.', 'Gated DeltaNet fast path needs the optional packages flash-linear-attention (fla) and causal-conv1d; without them the pure-torch implementation is used (modeling_qwen3_5.py:205-207). pip list of the spatialcraft env shows neither package.', 'README:608 says the latest transformers is required and documents only serving (transformers ser…

**language_model**

- model_type: qwen3_5_text
- num_layers: 64
- hidden_size: 5120
- num_attention_heads: 24
- num_key_value_heads: 4
- head_dim: 256
- layer_mixers（逐层，0 起）: `L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: linear_attention layers (indices i with (i+1)%4 != 0; 48 of 64 layers) are instances of Qwen3_5GatedDeltaNet (modeling_qwen3_5.py:356), built at modeling_qwen3_5.py:736-737. It is a Gated DeltaNet recurrent/linear-time mixer, not softmax attention: the README calls the block "Gated DeltaNet" (README:41). Mechanism from source: in_proj_qkv projects the hidden state to q,k,v; a depthwise causal Conv1d with kernel 4 (linear_conv_kernel_dim) + SiLU is applied to the concatenated qkv channels (modeling_qwen3_5.py:375-382); q and k are L2-normalised; a per-head matrix state S (key_dim x value_dim per head) is updated per position with the gated delta rule S_t = exp(g_t)*S_(t-1) + k_t (x) beta_t*(v_t - S k_t), output o_t = S_t q_t (reference loop modeling_qwen3_5.py:314-353; chunked parallel form used for full sequences modeling_qwen3_5.py:234, called at modeling_qwen3_5.py:499); beta = sigmoid(in_proj_b), g = -exp(A_log)*softplus(in_proj_a + dt_bias). The output is passed through a gated RMSNorm (norm, then multiply by SiLU(z) with z = in_proj_z(hidden)) and out_proj (modeling_qwen3_5.py:529-532). Heads: linear_num_key_heads=16 (q/k head dim 128), linear_num_value_heads=48 (value head d…
- mlp_type: dense: Qwen3_5MLP, SwiGLU-style gate_proj/up_proj/down_proj with SiLU, intermediate_size=17408 (modeling_qwen3_5.py:695-708, built at 740). No expert/MoE keys in config.json; README:41 lists FFN.
- head_level_analysis: Layers that admit standard head-level analysis: the 16 full_attention layers, indices [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63] (every 4th layer, (i+1)%4==0). Why: each has 24 softmax attention heads (Qwen3_5Attention); the per-head outputs are still separate (shape [batch, seq, 24*256], head h = slice h*256:(h+1)*256) at the input of self_attn.o_proj (modeling_qwen3_5.py:691), so a forward-pre-hook on o_proj reads or ablates single heads (that tensor is already multiplied by the sigmoid output gate, modeling_qwen3_5.py:689). Position-to-position attention masking is possible because these layers take a 4D causal mask built by create_causal_mask (modeling_qwen3_5.py:1260, selected at 1273) that is added to the logits before softmax in the eager path (modeling_qwen3_5.py:607-609). Attention weights require attn_implementation="eager"; under the default sdpa they are None. With eager, output_attentions yields one entry per full_attention layer only (hooks are installed on Qwen3_5Attention modules and None outputs are skipped: modeling_qwen3_5.py:795-799, utils/output_capturing.py:103-117), so the returned tuple has 16 entries and entry j corresponds to layer ind…

**vision_encoder**

- type: Qwen3_5VisionModel (modeling_qwen3_5.py:995): ViT with a Conv3d patch embedding (kernel/stride = temporal 2 x 16 x 16, modeling_qwen3_5.py:828-837), 27 pre-norm blocks with 16-head non-causal attention (modeling_qwen3_5.py:878), MLP width 4304, output projected to 5120 by the patch merger. The README only says "Vision Encoder" (README:34) and does not name the pretrained encoder family.
- num_layers: 27
- hidden_size: 1152
- patch_size: 16
- resolution_strategy: dynamic, single view. Image processor resizes each image with smart_resize(factor=32=patch 16 x merge 2): both sides rounded to multiples of 32, aspect ratio kept, total pixels clamped to [min_pixels, max_pixels] = [size.shortest_edge, size.longest_edge] = [65536, 16777216] (preprocessor_config.json; image_processing_qwen2_vl.py:62-88, 171-177). Aspect ratio above 200 raises. Resample is the class default BICUBIC (image_processing_qwen2_vl.py:94); mean/std 0.5. Default range is therefore 64 to 16384 LM tokens per image.
- position_encoding: Two schemes together. (1) Learned absolute table pos_embed = nn.Embedding(2304, 1152) treated as a 48x48 grid and bilinearly interpolated to the actual (grid_h, grid_w) of each image, then added to the patch embeddings (modeling_qwen3_5.py:1014-1015, 1070-1131). (2) 2D rotary embedding over (row, col) patch coordinates applied to q/k in every vision attention block (modeling_qwen3_5.py:1030-1068).

**connector**

- type: Qwen3_5VisionPatchMerger (modeling_qwen3_5.py:848-861): LayerNorm on each patch, concatenate the 4 patches of each 2x2 block, then Linear -> GELU -> Linear to out_hidden_size=5120. deepstack_visual_indexes is [] in config.json and the transformers 5.5.4 Qwen3.5 code has no deepstack path (removed in modular_qwen3_5.py), so visual features enter the LM only once, by replacing the <|image_pad|> input embeddings (modeling_qwen3_5.py:1648).
- token_merge: yes: spatial_merge_size=2, 2x2 patches -> 1 token (4x fewer tokens than patches); 32x32 resized pixels per LM token.
- tokens_per_image: N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216) is the resized size (multiples of 32). Equivalently image_grid_thw.prod()/4 with grid = (1, H2/16, W2/16) (processing_qwen3_vl.py:128-133; modeling_qwen3_5.py:1515). The chat template wraps them as <|vision_start|> + N x <|image_pad|> + <|vision_end|> (chat_template.jinja:18), i.e. N+2 sequence positions per image.
- token_to_region: Raster (row-major) order at the LM-token level. Let mw = W2/32, mh = H2/32. The k-th <|image_pad|> position of an image (k from 0) is merged cell row r = k // mw, col c = k % mw, covering resized-image pixels x in [32c, 32c+32), y in [32r, 32r+32); multiply by W/W2 and H/H2 to get original-image pixels. Reason: the image processor emits patches block-major, in order (block_row, block_col, intra_row, intra_col) (image_processing_qwen2_vl.py:203-220), the pos-embed is permuted the same way (modeling_qwen3_5.py:1125-1127), and the merger concatenates every 4 consecutive patches (modeling_qwen3_5.py:859), which are exactly one 2x2 block. Consequence for hooks INSIDE the vision encoder: patch index p is not raster; block b = p // 4, j = p % 4, patch row = 2*(b // mw) + j // 2, patch col = 2*(b % mw) + j % 2. A still image is duplicated along time to fill temporal_patch_size=2, giving grid_t = 1 (image_processing_qwen2_vl.py:196-200).

**multi_view**

- default: single view: no tiling, slicing or thumbnail. The image processor only resizes and patchifies each image into one grid (image_processing_qwen2_vl.py:166-232) and the processor inserts one placeholder run per image (processing_qwen3_vl.py:127-135).
- force_single_view: Nothing to turn off. To fix the token count, either pre-resize the image yourself to an exact (H2, W2) with both sides multiples of 32 and 65536 <= H2*W2 <= 16777216 (smart_resize then leaves it unchanged, giving exactly H2*W2/1024 tokens), or narrow the range by passing min_pixels and max_pixels together (both are needed for a per-call override, image_processing_qwen2_vl.py:132-133) or size={"shortest_edge": ..., "longest_edge": ...} to the processor / image processor. With min_pixels == max_pixels the count is only approximately fixed because of rounding to multiples of 32. Not run-tested here.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated locally. README.md was searched for grounding / bounding box / bbox / bbox_2d / point_2d / coordinate / normalized / 0-1000: the only hits are benchmark table rows (RefCOCO / ODInW) at README lines [420], with no prompt or coordinate convention. chat_template.jinja has no grounding instructions.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is ON by default. chat_template.jinja:149-153: when the template variable enable_thinking is defined and false the generation prompt ends with an empty block '<think>\n\n</think>\n\n'; otherwise it ends with an open '<think>\n'. To disable, set enable_thinking=False as a chat-template variable. README:764 states the model thinks by default; README:799 shows the serving form "chat_template_kwargs": {"enable_thinking": False}. README:762: the /think and /nothink soft switches are not officially supported. preserve_thinking (default off, chat_template.jinja:100) only affects how earlier assistant turns are rendered.
- source: chat_template.jinja:149-153; README.md:764, 799, 762

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- bbox.official_prompt / output_format / coord_range / coord_order: the local README.md contains no grounding or bounding-box prompt and no coordinate convention (only benchmark table rows at README lines [420]); chat_template.jinja and the processor source contain none either. Not imported from other Qwen generations.
- vision_encoder pretrained origin / family name: not stated in README or config; only the architecture is known from source.
- text_config keys that the transformers 5.5.4 modeling code never reads (grep of the modeling file), so their intended effect is unknown from local files: attn_output_gate, output_gate_type, mtp_num_hidden_layers, mtp_use_dedicated_embeddings, mamba_ssm_dtype, full_attention_interval (attn_output_gate: the gate is hard-coded on; full_attention_interval: only used when layer_types is absent; mtp weights are ignored on load).
- All head_level_analysis / output_attentions statements come from reading source; no forward pass was run (login node, no weights loaded).
- In-process loading example (AutoProcessor/AutoModel snippet) and passing enable_thinking via apply_chat_template: README shows only the serving API form.

### Qwen3.6-35B-A3B

- **hf_repo**: Qwen/Qwen3.6-35B-A3B
- **revision**: 995ad96eacd98c81ed38be0c5b274b04031597b0
- **revision_source**: MANIFEST.md line 11
- **size_bytes**: 71926865825
- **license**: apache-2.0
- **license_source**: README.md:3 front matter (license: apache-2.0); LICENSE:1-2 (Apache License, Version 2.0); MANIFEST.md line 11
- **role**: general
- **base_model**: —

**loader**

- model_class: Qwen3_5MoeForConditionalGeneration
- processor_class: Qwen3VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.57.1
- notes: ['model_type=qwen3_5_moe; config.json architectures=[Qwen3_5MoeForConditionalGeneration], which is also the AutoModelForImageTextToText mapping in transformers 5.5.4 (models/auto/modeling_auto.py:1017).', 'preprocessor_config.json names processor_class=Qwen3VLProcessor and image_processor_type=Qwen2VLImageProcessorFast; in transformers 5.5.4 the image-processor mapping for qwen3_5_moe is Qwen2VLImageProcessor (torchvision backend) with the legacy "Fast" name resolved to it (models/auto/image_processing_auto.py:219). Tokenizer mapping: Qwen3_5Tokenizer; video: Qwen3VLVideoProcessor.', 'trust_remote_code not needed: the model directory contains no .py files and tokenizer_config.json has no auto_map; all classes ship in the installed transformers 5.5.4 (spatialcraft env).', 'config.json was written by transformers 4.57.1; the code read for this file is 5.5.4.', 'Gated DeltaNet fast path needs the optional packages flash-linear-attention (fla) and causal-conv1d; without them the pure-torch implementation is used (modeling_qwen3_5_moe.py:206-208). pip list of the spatialcraft env shows neither package.', 'README:636 says the latest transformers is required and documents only serving (t…

**language_model**

- model_type: qwen3_5_moe_text
- num_layers: 40
- hidden_size: 2048
- num_attention_heads: 16
- num_key_value_heads: 2
- head_dim: 256
- layer_mixers（逐层，0 起）: `L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: linear_attention layers (indices i with (i+1)%4 != 0; 30 of 40 layers) are instances of Qwen3_5MoeGatedDeltaNet (modeling_qwen3_5_moe.py:357), built at modeling_qwen3_5_moe.py:820-821. It is a Gated DeltaNet recurrent/linear-time mixer, not softmax attention: the README calls the block "Gated DeltaNet" (README:41). Mechanism from source: in_proj_qkv projects the hidden state to q,k,v; a depthwise causal Conv1d with kernel 4 (linear_conv_kernel_dim) + SiLU is applied to the concatenated qkv channels (modeling_qwen3_5_moe.py:376-383); q and k are L2-normalised; a per-head matrix state S (key_dim x value_dim per head) is updated per position with the gated delta rule S_t = exp(g_t)*S_(t-1) + k_t (x) beta_t*(v_t - S k_t), output o_t = S_t q_t (reference loop modeling_qwen3_5_moe.py:315-354; chunked parallel form used for full sequences modeling_qwen3_5_moe.py:235, called at modeling_qwen3_5_moe.py:500); beta = sigmoid(in_proj_b), g = -exp(A_log)*softplus(in_proj_a + dt_bias). The output is passed through a gated RMSNorm (norm, then multiply by SiLU(z) with z = in_proj_z(hidden)) and out_proj (modeling_qwen3_5_moe.py:530-533). Heads: linear_num_key_heads=16 (q/k head dim 128), linear_n…
- mlp_type: moe: Qwen3_5MoeSparseMoeBlock in every layer (modeling_qwen3_5_moe.py:824, class at 773). num_experts=256 routed experts, num_experts_per_tok=8 selected per token by a softmax-then-top-k router (modeling_qwen3_5_moe.py:754-770), expert intermediate size moe_intermediate_size=512; plus 1 always-on shared expert (a SwiGLU MLP with shared_expert_intermediate_size=512) whose output is scaled by sigmoid(shared_expert_gate(h)) and added (modeling_qwen3_5_moe.py:778-779, 788-790). README:51: "8 Routed + 1 Shared". Router outputs can be captured with output_router_logits (modeling_qwen3_5_moe.py:883).
- head_level_analysis: Layers that admit standard head-level analysis: the 10 full_attention layers, indices [3, 7, 11, 15, 19, 23, 27, 31, 35, 39] (every 4th layer, (i+1)%4==0). Why: each has 16 softmax attention heads (Qwen3_5MoeAttention); the per-head outputs are still separate (shape [batch, seq, 16*256], head h = slice h*256:(h+1)*256) at the input of self_attn.o_proj (modeling_qwen3_5_moe.py:694), so a forward-pre-hook on o_proj reads or ablates single heads (that tensor is already multiplied by the sigmoid output gate, modeling_qwen3_5_moe.py:692). Position-to-position attention masking is possible because these layers take a 4D causal mask built by create_causal_mask (modeling_qwen3_5_moe.py:1385, selected at 1398) that is added to the logits before softmax in the eager path (modeling_qwen3_5_moe.py:608-610). Attention weights require attn_implementation="eager"; under the default sdpa they are None. With eager, output_attentions yields one entry per full_attention layer only (hooks are installed on Qwen3_5MoeAttention modules and None outputs are skipped: modeling_qwen3_5_moe.py:882-886, utils/output_capturing.py:103-117), so the returned tuple has 10 entries and entry j corresponds to layer i…

**vision_encoder**

- type: Qwen3_5MoeVisionModel (modeling_qwen3_5_moe.py:1088): ViT with a Conv3d patch embedding (kernel/stride = temporal 2 x 16 x 16, modeling_qwen3_5_moe.py:921-930), 27 pre-norm blocks with 16-head non-causal attention (modeling_qwen3_5_moe.py:971), MLP width 4304, output projected to 2048 by the patch merger. The README only says "Vision Encoder" (README:34) and does not name the pretrained encoder family.
- num_layers: 27
- hidden_size: 1152
- patch_size: 16
- resolution_strategy: dynamic, single view. Image processor resizes each image with smart_resize(factor=32=patch 16 x merge 2): both sides rounded to multiples of 32, aspect ratio kept, total pixels clamped to [min_pixels, max_pixels] = [size.shortest_edge, size.longest_edge] = [65536, 16777216] (preprocessor_config.json; image_processing_qwen2_vl.py:62-88, 171-177). Aspect ratio above 200 raises. Resample is the class default BICUBIC (image_processing_qwen2_vl.py:94); mean/std 0.5. Default range is therefore 64 to 16384 LM tokens per image.
- position_encoding: Two schemes together. (1) Learned absolute table pos_embed = nn.Embedding(2304, 1152) treated as a 48x48 grid and bilinearly interpolated to the actual (grid_h, grid_w) of each image, then added to the patch embeddings (modeling_qwen3_5_moe.py:1107-1108, 1163-1224). (2) 2D rotary embedding over (row, col) patch coordinates applied to q/k in every vision attention block (modeling_qwen3_5_moe.py:1123-1161).

**connector**

- type: Qwen3_5MoeVisionPatchMerger (modeling_qwen3_5_moe.py:941-954): LayerNorm on each patch, concatenate the 4 patches of each 2x2 block, then Linear -> GELU -> Linear to out_hidden_size=2048. deepstack_visual_indexes is [] in config.json and the transformers 5.5.4 Qwen3.5 code has no deepstack path (removed in modular_qwen3_5.py), so visual features enter the LM only once, by replacing the <|image_pad|> input embeddings (modeling_qwen3_5_moe.py:1773).
- token_merge: yes: spatial_merge_size=2, 2x2 patches -> 1 token (4x fewer tokens than patches); 32x32 resized pixels per LM token.
- tokens_per_image: N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216) is the resized size (multiples of 32). Equivalently image_grid_thw.prod()/4 with grid = (1, H2/16, W2/16) (processing_qwen3_vl.py:128-133; modeling_qwen3_5_moe.py:1640). The chat template wraps them as <|vision_start|> + N x <|image_pad|> + <|vision_end|> (chat_template.jinja:18), i.e. N+2 sequence positions per image.
- token_to_region: Raster (row-major) order at the LM-token level. Let mw = W2/32, mh = H2/32. The k-th <|image_pad|> position of an image (k from 0) is merged cell row r = k // mw, col c = k % mw, covering resized-image pixels x in [32c, 32c+32), y in [32r, 32r+32); multiply by W/W2 and H/H2 to get original-image pixels. Reason: the image processor emits patches block-major, in order (block_row, block_col, intra_row, intra_col) (image_processing_qwen2_vl.py:203-220), the pos-embed is permuted the same way (modeling_qwen3_5_moe.py:1218-1220), and the merger concatenates every 4 consecutive patches (modeling_qwen3_5_moe.py:952), which are exactly one 2x2 block. Consequence for hooks INSIDE the vision encoder: patch index p is not raster; block b = p // 4, j = p % 4, patch row = 2*(b // mw) + j // 2, patch col = 2*(b % mw) + j % 2. A still image is duplicated along time to fill temporal_patch_size=2, giving grid_t = 1 (image_processing_qwen2_vl.py:196-200).

**multi_view**

- default: single view: no tiling, slicing or thumbnail. The image processor only resizes and patchifies each image into one grid (image_processing_qwen2_vl.py:166-232) and the processor inserts one placeholder run per image (processing_qwen3_vl.py:127-135).
- force_single_view: Nothing to turn off. To fix the token count, either pre-resize the image yourself to an exact (H2, W2) with both sides multiples of 32 and 65536 <= H2*W2 <= 16777216 (smart_resize then leaves it unchanged, giving exactly H2*W2/1024 tokens), or narrow the range by passing min_pixels and max_pixels together (both are needed for a per-call override, image_processing_qwen2_vl.py:132-133) or size={"shortest_edge": ..., "longest_edge": ...} to the processor / image processor. With min_pixels == max_pixels the count is only approximately fixed because of rounding to multiples of 32. Not run-tested here.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated locally. README.md was searched for grounding / bounding box / bbox / bbox_2d / point_2d / coordinate / normalized / 0-1000: the only hits are benchmark table rows (RefCOCO / ODInW) at README lines [443, 452], with no prompt or coordinate convention. chat_template.jinja has no grounding instructions.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is ON by default. chat_template.jinja:149-153: when the template variable enable_thinking is defined and false the generation prompt ends with an empty block '<think>\n\n</think>\n\n'; otherwise it ends with an open '<think>\n'. To disable, set enable_thinking=False as a chat-template variable. README:792 states the model thinks by default; README:827 shows the serving form "chat_template_kwargs": {"enable_thinking": False}. README:790: the /think and /nothink soft switches are not officially supported. preserve_thinking (default off, chat_template.jinja:100) only affects how earlier assistant turns are rendered.
- source: chat_template.jinja:149-153; README.md:792, 827, 790

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- bbox.official_prompt / output_format / coord_range / coord_order: the local README.md contains no grounding or bounding-box prompt and no coordinate convention (only benchmark table rows at README lines [443, 452]); chat_template.jinja and the processor source contain none either. Not imported from other Qwen generations.
- vision_encoder pretrained origin / family name: not stated in README or config; only the architecture is known from source.
- text_config keys that the transformers 5.5.4 modeling code never reads (grep of the modeling file), so their intended effect is unknown from local files: attn_output_gate, mtp_num_hidden_layers, mtp_use_dedicated_embeddings, mamba_ssm_dtype, full_attention_interval (attn_output_gate: the gate is hard-coded on; full_attention_interval: only used when layer_types is absent; mtp weights are ignored on load).
- All head_level_analysis / output_attentions statements come from reading source; no forward pass was run (login node, no weights loaded).
- In-process loading example (AutoProcessor/AutoModel snippet) and passing enable_thinking via apply_chat_template: README shows only the serving API form.

### Qwen3.8-27B

- **hf_repo**: Qwen/Qwen3.8-27B
- **revision**: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
- **revision_source**: MANIFEST.md line 49-56
- **size_bytes**: 55586114863
- **license**: apache-2.0
- **license_source**: README.md:3 front matter (license: apache-2.0); LICENSE:1-2 (Apache License, Version 2.0); MANIFEST.md line 49-56
- **role**: general
- **base_model**: —

**loader**

- model_class: Qwen3_5ForConditionalGeneration
- processor_class: Qwen3VLProcessor
- trust_remote_code: False
- config_transformers_version: 5.8.0.dev0
- notes: ['model_type=qwen3_5; config.json architectures=[Qwen3_5ForConditionalGeneration], which is also the AutoModelForImageTextToText mapping in transformers 5.5.4 (models/auto/modeling_auto.py:1016).', 'preprocessor_config.json names processor_class=Qwen3VLProcessor and image_processor_type=Qwen2VLImageProcessorFast; in transformers 5.5.4 the image-processor mapping for qwen3_5 is Qwen2VLImageProcessor (torchvision backend) with the legacy "Fast" name resolved to it (models/auto/image_processing_auto.py:218). Tokenizer mapping: Qwen3_5Tokenizer; video: Qwen3VLVideoProcessor.', 'trust_remote_code not needed: the model directory contains no .py files and tokenizer_config.json has no auto_map; all classes ship in the installed transformers 5.5.4 (spatialcraft env).', 'config.json was written by transformers 5.8.0.dev0; the code read for this file is 5.5.4.', 'Gated DeltaNet fast path needs the optional packages flash-linear-attention (fla) and causal-conv1d; without them the pure-torch implementation is used (modeling_qwen3_5.py:205-207). pip list of the spatialcraft env shows neither package.', 'MANIFEST.md:70: compared with Qwen3.6-27B in the same root, config.json differs only in tran…

**language_model**

- model_type: qwen3_5_text
- num_layers: 64
- hidden_size: 5120
- num_attention_heads: 24
- num_key_value_heads: 4
- head_dim: 256
- layer_mixers（逐层，0 起）: `L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F L L L F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: linear_attention layers (indices i with (i+1)%4 != 0; 48 of 64 layers) are instances of Qwen3_5GatedDeltaNet (modeling_qwen3_5.py:356), built at modeling_qwen3_5.py:736-737. It is a Gated DeltaNet recurrent/linear-time mixer, not softmax attention: the README calls the block "Gated DeltaNet" (README:41). Mechanism from source: in_proj_qkv projects the hidden state to q,k,v; a depthwise causal Conv1d with kernel 4 (linear_conv_kernel_dim) + SiLU is applied to the concatenated qkv channels (modeling_qwen3_5.py:375-382); q and k are L2-normalised; a per-head matrix state S (key_dim x value_dim per head) is updated per position with the gated delta rule S_t = exp(g_t)*S_(t-1) + k_t (x) beta_t*(v_t - S k_t), output o_t = S_t q_t (reference loop modeling_qwen3_5.py:314-353; chunked parallel form used for full sequences modeling_qwen3_5.py:234, called at modeling_qwen3_5.py:499); beta = sigmoid(in_proj_b), g = -exp(A_log)*softplus(in_proj_a + dt_bias). The output is passed through a gated RMSNorm (norm, then multiply by SiLU(z) with z = in_proj_z(hidden)) and out_proj (modeling_qwen3_5.py:529-532). Heads: linear_num_key_heads=16 (q/k head dim 128), linear_num_value_heads=48 (value head d…
- mlp_type: dense: Qwen3_5MLP, SwiGLU-style gate_proj/up_proj/down_proj with SiLU, intermediate_size=17408 (modeling_qwen3_5.py:695-708, built at 740). No expert/MoE keys in config.json; README:41 lists FFN.
- head_level_analysis: Layers that admit standard head-level analysis: the 16 full_attention layers, indices [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63] (every 4th layer, (i+1)%4==0). Why: each has 24 softmax attention heads (Qwen3_5Attention); the per-head outputs are still separate (shape [batch, seq, 24*256], head h = slice h*256:(h+1)*256) at the input of self_attn.o_proj (modeling_qwen3_5.py:691), so a forward-pre-hook on o_proj reads or ablates single heads (that tensor is already multiplied by the sigmoid output gate, modeling_qwen3_5.py:689). Position-to-position attention masking is possible because these layers take a 4D causal mask built by create_causal_mask (modeling_qwen3_5.py:1260, selected at 1273) that is added to the logits before softmax in the eager path (modeling_qwen3_5.py:607-609). Attention weights require attn_implementation="eager"; under the default sdpa they are None. With eager, output_attentions yields one entry per full_attention layer only (hooks are installed on Qwen3_5Attention modules and None outputs are skipped: modeling_qwen3_5.py:795-799, utils/output_capturing.py:103-117), so the returned tuple has 16 entries and entry j corresponds to layer ind…

**vision_encoder**

- type: Qwen3_5VisionModel (modeling_qwen3_5.py:995): ViT with a Conv3d patch embedding (kernel/stride = temporal 2 x 16 x 16, modeling_qwen3_5.py:828-837), 27 pre-norm blocks with 16-head non-causal attention (modeling_qwen3_5.py:878), MLP width 4304, output projected to 5120 by the patch merger. The README only says "Vision Encoder" (README:34) and does not name the pretrained encoder family.
- num_layers: 27
- hidden_size: 1152
- patch_size: 16
- resolution_strategy: dynamic, single view. Image processor resizes each image with smart_resize(factor=32=patch 16 x merge 2): both sides rounded to multiples of 32, aspect ratio kept, total pixels clamped to [min_pixels, max_pixels] = [size.shortest_edge, size.longest_edge] = [65536, 16777216] (preprocessor_config.json; image_processing_qwen2_vl.py:62-88, 171-177). Aspect ratio above 200 raises. Resample is the class default BICUBIC (image_processing_qwen2_vl.py:94); mean/std 0.5. Default range is therefore 64 to 16384 LM tokens per image.
- position_encoding: Two schemes together. (1) Learned absolute table pos_embed = nn.Embedding(2304, 1152) treated as a 48x48 grid and bilinearly interpolated to the actual (grid_h, grid_w) of each image, then added to the patch embeddings (modeling_qwen3_5.py:1014-1015, 1070-1131). (2) 2D rotary embedding over (row, col) patch coordinates applied to q/k in every vision attention block (modeling_qwen3_5.py:1030-1068).

**connector**

- type: Qwen3_5VisionPatchMerger (modeling_qwen3_5.py:848-861): LayerNorm on each patch, concatenate the 4 patches of each 2x2 block, then Linear -> GELU -> Linear to out_hidden_size=5120. deepstack_visual_indexes is [] in config.json and the transformers 5.5.4 Qwen3.5 code has no deepstack path (removed in modular_qwen3_5.py), so visual features enter the LM only once, by replacing the <|image_pad|> input embeddings (modeling_qwen3_5.py:1648).
- token_merge: yes: spatial_merge_size=2, 2x2 patches -> 1 token (4x fewer tokens than patches); 32x32 resized pixels per LM token.
- tokens_per_image: N = (H2/32) * (W2/32), where (H2, W2) = smart_resize(H, W, factor=32, min_pixels=65536, max_pixels=16777216) is the resized size (multiples of 32). Equivalently image_grid_thw.prod()/4 with grid = (1, H2/16, W2/16) (processing_qwen3_vl.py:128-133; modeling_qwen3_5.py:1515). The chat template wraps them as <|vision_start|> + N x <|image_pad|> + <|vision_end|> (chat_template.jinja:18), i.e. N+2 sequence positions per image.
- token_to_region: Raster (row-major) order at the LM-token level. Let mw = W2/32, mh = H2/32. The k-th <|image_pad|> position of an image (k from 0) is merged cell row r = k // mw, col c = k % mw, covering resized-image pixels x in [32c, 32c+32), y in [32r, 32r+32); multiply by W/W2 and H/H2 to get original-image pixels. Reason: the image processor emits patches block-major, in order (block_row, block_col, intra_row, intra_col) (image_processing_qwen2_vl.py:203-220), the pos-embed is permuted the same way (modeling_qwen3_5.py:1125-1127), and the merger concatenates every 4 consecutive patches (modeling_qwen3_5.py:859), which are exactly one 2x2 block. Consequence for hooks INSIDE the vision encoder: patch index p is not raster; block b = p // 4, j = p % 4, patch row = 2*(b // mw) + j // 2, patch col = 2*(b % mw) + j % 2. A still image is duplicated along time to fill temporal_patch_size=2, giving grid_t = 1 (image_processing_qwen2_vl.py:196-200).

**multi_view**

- default: single view: no tiling, slicing or thumbnail. The image processor only resizes and patchifies each image into one grid (image_processing_qwen2_vl.py:166-232) and the processor inserts one placeholder run per image (processing_qwen3_vl.py:127-135).
- force_single_view: Nothing to turn off. To fix the token count, either pre-resize the image yourself to an exact (H2, W2) with both sides multiples of 32 and 65536 <= H2*W2 <= 16777216 (smart_resize then leaves it unchanged, giving exactly H2*W2/1024 tokens), or narrow the range by passing min_pixels and max_pixels together (both are needed for a per-call override, image_processing_qwen2_vl.py:132-133) or size={"shortest_edge": ..., "longest_edge": ...} to the processor / image processor. With min_pixels == max_pixels the count is only approximately fixed because of rounding to multiples of 32. Not run-tested here.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated locally. README.md was searched for grounding / bounding box / bbox / bbox_2d / point_2d / coordinate / normalized / 0-1000: no hits at all. chat_template.jinja has no grounding instructions.

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is ON by default. chat_template.jinja:165-169: when the template variable enable_thinking is defined and false the generation prompt ends with an empty block '<think>\n\n</think>\n\n'; otherwise it ends with an open '<think>\n'. To disable, set enable_thinking=False as a chat-template variable. README:425 states the model thinks by default; README:459 shows the serving form "chat_template_kwargs": {"enable_thinking": False}. Qwen3.8-specific: while thinking is enabled the template also resolves reasoning_effort (default xhigh; allowed xhigh/medium/low, anything else raises) and, for xhigh and low, puts an instruction sentence ("Reasoning effort is set to ...") at the start of the system turn, creating a system turn when the conversation has none (chat_template.jinja:46-56, 76-86; README:258). So with default settings the prompt contains a system message the user did not write; with enable_thinking=False no such sentence is added. preserve_thinking defaults to on (chat_template.jinja:116) and only affects how earlier assistant turns are rendered.
- source: chat_template.jinja:165-169; README.md:425, 459

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- bbox.official_prompt / output_format / coord_range / coord_order: the local README.md contains no grounding or bounding-box prompt and no coordinate convention (no match at all for grounding/bbox/coordinate terms); chat_template.jinja and the processor source contain none either. Not imported from other Qwen generations.
- vision_encoder pretrained origin / family name: not stated in README or config; only the architecture is known from source.
- text_config keys that the transformers 5.5.4 modeling code never reads (grep of the modeling file), so their intended effect is unknown from local files: attn_output_gate, output_gate_type, mtp_num_hidden_layers, mtp_use_dedicated_embeddings, mamba_ssm_dtype, full_attention_interval (attn_output_gate: the gate is hard-coded on; full_attention_interval: only used when layer_types is absent; mtp weights are ignored on load).
- All head_level_analysis / output_attentions statements come from reading source; no forward pass was run (login node, no weights loaded).
- In-process loading example (AutoProcessor/AutoModel snippet) and passing enable_thinking via apply_chat_template: README shows only the serving API form.
- Whether Qwen3.8-27B weights differ from Qwen3.6-27B in values: MANIFEST only compared tensor names/shapes/dtypes.

### RoboBrain2.0-7B

- **hf_repo**: BAAI/RoboBrain2.0-7B
- **revision**: f844fdfeab868f4ebdc7edc55aa66b907ab2a57d
- **revision_source**: _logs/hf_download_394207.out line 1 (successful run; same sha in the failed first attempt _logs/hf_download_394195.out). Not listed in MANIFEST.md.
- **size_bytes**: 16595974434
- **license**: apache-2.0
- **license_source**: README.md front matter line 7 (front matter also carries extra_gated_prompt / extra_gated_fields, lines 2-5)
- **role**: spatial_finetune
- **base_model**: —

**loader**

- model_class: Qwen2_5_VLForConditionalGeneration
- processor_class: Qwen2_5_VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.51.3
- notes: Native transformers class (auto mapping for model_type qwen2_5_vl); no custom code files in the repo. The model card gives no transformers snippet and no version requirement; it says "Please refer to RoboBrain 2.0 Github for the usage" (https://github.com/FlagOpen/RoboBrain2.0, pip install -r requirements.txt there). torch_dtype bfloat16. preprocessor_config.json names image_processor_type Qwen2VLImageProcessor; no video_preprocessor_config.json in the dir. generation_config.json ships do_sample=true, temperature=1e-06, repetition_penalty=1.05 (pass do_sample=False / repetition_penalty=1.0 explicitly if exact greedy decoding is wanted). Not covered by the MANIFEST.md smoke test (downloaded later), so loading under transformers 5.5.4 is untested.

**language_model**

- model_type: qwen2_5_vl_text
- num_layers: 28
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: All 28 layers are plain causal self-attention. config.json has sliding_window=32768 and max_window_layers=28 but use_sliding_window=false; Qwen2_5_VLTextConfig.__post_init__ sets sliding_window=None when use_sliding_window is false and then builds layer_types as "sliding_attention" only if sliding_window is not None and i >= max_window_layers, so every layer is "full_attention" (and even with the switch on, i >= 28 never holds for 28 layers). Grouped-query attention: 28 query heads, 4 key/value heads (each kv head shared by 7 query heads). Position encoding in the LM is multimodal RoPE with mrope_section [16, 24, 24], rope_theta 1e6.
- mlp_type: dense
- head_level_analysis: All 28 layers have the same standard attention structure (q_proj -> 28 heads x 128 dims, o_proj takes the concatenated 28 heads), so head-level analysis / attention masking applies to every layer. Note k/v are shared within groups of 7 query heads (4 kv heads). Whether per-head attention weights are returned under the sdpa/flash backends was not checked here.

**vision_encoder**

- type: Qwen2.5-VL ViT (Qwen2_5_VisionTransformerPretrainedModel) with window attention
- num_layers: 32
- hidden_size: 1280
- patch_size: 14
- resolution_strategy: dynamic (native aspect ratio). Image processor smart_resize: round H and W to multiples of 28 (= patch 14 x merge 2); if H*W > max_pixels scale down (floor to multiple of 28), if H*W < min_pixels scale up (ceil to multiple of 28); aspect ratio > 200 raises. Values shipped in this model dir's preprocessor_config.json: min_pixels=3136 (56x56 -> 4 tokens), max_pixels=12845056 (= 16384 x 28 x 28 -> up to 16384 tokens). Resample bicubic, CLIP mean/std. The model card states no recommended pixel range.
- position_encoding: 2D rotary position embedding (RoPE) on (row, col) patch indices inside every vision block (rot_pos_emb; no learned absolute position table). Window attention: 28 of 32 blocks attend only inside 112x112 px windows (= 8x8 patches = 4x4 merged cells); blocks 7, 15, 23, 31 (fullatt_block_indexes) attend over the whole image. 16 heads, temporal_patch_size 2 (a still image is duplicated to 2 frames and embedded by a Conv3d).

**connector**

- type: Qwen2_5_VLPatchMerger: RMSNorm, concatenate each 2x2 group of patch features (4 x 1280 = 5120), then MLP Linear(5120,5120) -> GELU -> Linear(5120, 3584)
- token_merge: 2x2 spatial merge (4 patches -> 1 token), spatial_merge_size 2
- tokens_per_image: (H'/28) * (W'/28), where (H', W') is the smart_resize output (both multiples of 28). Processor: image_grid_thw.prod() // merge_size**2 with grid_t = 1 for a still image. The tokens are <|image_pad|> placeholders wrapped as <|vision_start|> ... <|vision_end|> by the chat template. With shipped limits: 4 to 16384 tokens per image.
- token_to_region: The tokens that reach the LM ARE in raster (row-major) order of merged 28x28 px cells. Details: the image processor orders patches as (merged_row, merged_col, 2x2 inside the cell), the vision tower then permutes merged cells into window order (window_index) for window attention, the merger fuses each consecutive group of 4 patches, and finally merged_hidden_states[argsort(window_index)] restores the original order before the features are masked_scatter-ed into the <|image_pad|> positions. Rule: with gw = W'/28, the k-th image token (0-based) is row r = k // gw, col c = k % gw and covers resized-image pixels x in [28c, 28c+28), y in [28r, 28r+28); in original-image pixels multiply x by W/W' and y by H/H'. CAVEAT for vision-tower internals: the pre-merger hidden states (last_hidden_state / per-block outputs) are left in WINDOW order, not raster order; only the merged pooler_output is un-permuted, so apply argsort(window_index) yourself when reading intermediate ViT activations.

**multi_view**

- default: Single view. No tiling, no thumbnail, no newline tokens; one native-resolution grid per image.
- force_single_view: Nothing to disable. To fix the token count, pass min_pixels / max_pixels (processor or image-processor kwargs) or pre-resize the image to multiples of 28.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md:64 ("spatial perception for precise point and bbox prediction from complex instructions") and README.md:72; README.md:103 defers usage to the GitHub repo. No task prompts (pointing / affordance / trajectory / grounding) are present in any local file.

**thinking**

- has_thinking_mode: —
- how_to_disable: —
- source: README.md:72 mentions "long-chain-of-thought reasoning" as a capability, but neither the card nor chat_template.json contains an enable_thinking flag or a thinking prompt. chat_template.json is the plain Qwen2.5-VL template (default system prompt "You are a helpful assistant.").

**查不到 / 未确认**

- base_model: the model card never names the base checkpoint (e.g. whether it is Qwen2.5-VL-7B-Instruct); only the architecture (qwen2_5_vl) is evident from config.json.
- bbox.official_prompt / output_format / coord_range / coord_order: the card contains no prompts or coordinate convention; the official task prompts live in the GitHub repo (https://github.com/FlagOpen/RoboBrain2.0), which was not read.
- thinking: no enable_thinking switch or thinking prompt in the card or chat template; whether the GitHub inference code has one is not known from local files.
- license gating: front matter has extra_gated_prompt/fields; MANIFEST.md does not list this model, so gated status at download time is not recorded locally.
- Loading under transformers 5.5.4 not smoke-tested per local records (config written by 4.51.3).

### SenseNova-SI-1.1-InternVL3-2B

- **hf_repo**: sensenova/SenseNova-SI-1.1-InternVL3-2B
- **revision**: 81c0e4c513cb44a6f6defc8b03983aa97ab32567
- **revision_source**: _logs/hf_download_394169.out line 4 (`下载 sensenova/SenseNova-SI-1.1-InternVL3-2B@81c0e4c5...`), verified line 5-6 (files=20 ok=True). Not listed in MANIFEST.md.
- **size_bytes**: 4186841742
- **license**: apache-2.0
- **license_source**: README.md front matter line 4 (`license: apache-2.0`). Not listed in MANIFEST.md.
- **role**: spatial_finetune
- **base_model**: OpenGVLab/InternVL3-2B - from README.md front matter lines 2-3: `base_model:` / `- OpenGVLab/InternVL3-2B`. The README body does not name the base checkpoint in a sentence; the closest statements are: "we explore scaling up multimodal foundation models to cultivate spatial intelligence within the **SenseNova-SI family**, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel)." and "Currently, we build SenseNova-SI upon popular open-source foundation models to maximize comp…

**loader**

- model_class: AutoModel (config.json auto_map -> "sensenova/SenseNova-SI-1.1-InternVL3-2B--modeling_internvl_chat.InternVLChatModel"; architectures: InternVLChatModel)
- processor_class: No processor class. Tokenizer: tokenizer_config.json tokenizer_class Qwen2Tokenizer (NO tokenizer.json in this directory - only vocab.json + merges.txt + added_tokens.json). Image preprocessing is done by the caller; the model card contains no preprocessing code (it delegates to example.py of the GitHub repo OpenSenseNova/SenseNova-SI, which is not in this directory). No preprocessor_config.json in this directory.
- trust_remote_code: True
- config_transformers_version: 4.47.1
- notes: - config.json top-level `transformers_version` is null (line 117); "4.47.1" is the value recorded inside llm_config (line 100), vision_config (line 216) and generation_config.json. - No transformers version pin appears in README.md / README_CN.md. Installation is described only as `git clone git@github.com:OpenSenseNova/SenseNova-SI.git; uv sync --extra cu124; uv sync` (the pinned versions live in that repo's lock file, not here). The remote code asserts transformers >= 4.37.0 (modeling_internvl_chat.py:46). - Usage in the card is only via `python example.py --image_paths ... --question ... --model_path sensenova/SenseNova-SI-1.1-InternVL3-2B` (example.py is not shipped here). - auto_map entries are repo-prefixed ("sensenova/SenseNova-SI-1.1-InternVL3-2B--configuration_internvl_chat.InternVLChatConfig" etc., config.json lines 7-11). In transformers 5.5.4, get_class_from_dynamic_module takes the repo id from the part before "--" (dynamic_module_utils.py:563-566) and fetches the module from that repo, so loading this local directory with AutoModel/AutoConfig may try to resolve the code from the Hub / HF cache instead of the .py files in the directory. Not tested. - Remote code suppo…

**language_model**

- model_type: qwen2
- num_layers: 28
- hidden_size: 1536
- num_attention_heads: 12
- num_key_value_heads: 2
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: All 28 layers are standard causal full self-attention with grouped-query attention (12 query heads, 2 KV heads). head_dim 128 is computed, not stated: config.json llm_config has no head_dim key; transformers Qwen2Attention uses getattr(config, "head_dim", hidden_size // num_attention_heads) = 1536 // 12 = 128 (models/qwen2/modeling_qwen2.py:195, transformers 5.5.4). No sliding window is active: llm_config has use_sliding_window: false and sliding_window: null (max_window_layers: 70 > 28 layers and inert). In transformers 5.5.4 Qwen2Config.__post_init__ sets sliding_window=None when use_sliding_window is false and builds layer_types as all "full_attention" (models/qwen2/configuration_qwen2.py:84-94). rope_scaling in llm_config: {factor: 2.0, rope_type: dynamic, type: dynamic}, rope_theta 1000000.0. llm_config._name_or_path is "./pretrained/Qwen2.5-32B-Instruct", which does not match the dimensions (hidden 1536, 28 layers) - treat it as a stale path string, not as the LLM identity.
- mlp_type: dense (architectures = Qwen2ForCausalLM; hidden_act silu, intermediate_size 8960; llm_config.moe_config is null)
- head_level_analysis: All 28 layers have the same structure (12 heads), so head-level analysis is structurally possible on every layer. Precondition: eager attention in the LLM. config.json asks for flash_attention_2 and the remote code re-forces it when use_flash_attn=True and flash_attn is importable (modeling_internvl_chat.py:55-57); load with use_flash_attn=False so the code sets 'eager'. In transformers 5.5.4 only eager_attention_forward returns the softmaxed weights (models/qwen2/modeling_qwen2.py:161-183) and config.output_attentions raises unless _attn_implementation is 'eager' (configuration_utils.py:346-351). InternVLChatModel.forward passes output_attentions to the language model and returns outputs.attentions (modeling_internvl_chat.py:137,165). Head masking is not offered by an argument in the code read; it would need hooks. Not tested (no weights were loaded). Vision encoder: InternAttention never returns attention weights (_naive_attn returns only x; FlashAttention asserts not need_weights), so ViT attention needs hooks / re-computation.

**vision_encoder**

- type: InternVisionModel from the shipped modeling_intern_vit.py. config.json vision_config.model_type is "intern_vit_6b" and vision_config._name_or_path is "OpenGVLab/InternViT-6B-448px-V1-5", but the actual dimensions are hidden_size 1024, 24 layers, 16 heads, intermediate_size 4096, norm_type layer_norm, qkv_bias true (i.e. not 6B-sized; the two strings contradict the numbers). The README does not name the vision encoder. vision_config also carries MoE-related keys (num_experts 8, use_moe false, ...) that the shipped modeling_intern_vit.py does not read.
- num_layers: 24
- hidden_size: 1024
- patch_size: 14
- resolution_strategy: Encoder input is fixed 448x448 per tile (vision_config.image_size 448, force_image_size 448) -> 32x32 = 1024 patch tokens + 1 CLS. Higher resolution is handled outside the encoder by dynamic tiling: config dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true (see multi_view).
- position_encoding: Learned absolute position embedding: nn.Parameter of shape (1, 1025, 1024) = CLS + 32x32 (modeling_intern_vit.py:149-152). At forward time the patch part goes through _get_pos_embed, which bicubic-interpolates (F.interpolate, mode='bicubic', align_corners=False) the 32x32 grid to the actual (H, W) patch grid (lines 154-173); at 448x448 input this is the identity size. No rotary / 2D relative encoding in the ViT.

**connector**

- type: pixel_shuffle (space-to-depth, downsample_ratio 0.5, ps_version v2) followed by `mlp1` = Sequential(LayerNorm(4096), Linear(4096, 1536), GELU(), Linear(1536, 1536)) (modeling_internvl_chat.py:78-83; 4096 = vit_hidden 1024 * int(1/0.5)^2; llm hidden 1536). Uses the last ViT layer (select_layer -1), CLS token dropped.
- token_merge: 2x2 patch merge by pixel_shuffle (4 ViT patch tokens -> 1 LLM token, channels concatenated 1024 -> 4096); 4x token reduction
- tokens_per_image: 256 per 448x448 tile: num_image_token = int((448 // 14) ** 2 * (0.5 ** 2)) = int(1024 * 0.25) = 256 (modeling_internvl_chat.py:52). Per image: 256 * num_tiles. In the prompt each image becomes '<img>' + '<IMG_CONTEXT>' * 256 * num_tiles + '</img>' (modeling_internvl_chat.py:284), i.e. one <img>...</img> pair per image, not per tile. Token ids (added_tokens.json): <img> 151665, </img> 151666, <IMG_CONTEXT> 151667.
- token_to_region: Derived from the shipped remote code (extract_feature + pixel_shuffle, ps_version=v2), not from a doc statement. 1) The ViT output drops the CLS token (vit_embeds[:, 1:, :]) and is reshaped to (tiles, 32, 32, 1024), row-major: dim1 = patch row (y), dim2 = patch column (x). 2) pixel_shuffle(scale_factor=0.5): view -> merges horizontally adjacent patch pairs into channels; permute(0,2,1,3); view -> merges vertically adjacent pairs; for ps_version != 'v1' a final permute(0,2,1,3) restores (row, col) order. Result (tiles, 16, 16, 4096). 3) reshape(tiles, -1, 4096) flattens row-major, so within one 448x448 tile, token k (0..255) has row i = k // 16, col j = k % 16 and covers ViT patches rows 2i..2i+1, cols 2j..2j+1, i.e. tile pixels x in [28*j, 28*j+28), y in [28*i, 28*i+28). Channel concat order inside a token: patch(2i,2j), patch(2i,2j+1), patch(2i+1,2j), patch(2i+1,2j+1). 4) Across tiles: the n-th <IMG_CONTEXT> position of an image (n counted from 0) maps to tile t = n // 256, k = n % 256, because vit_embeds.reshape(-1, C) is written into the IMG_CONTEXT positions in order. Tile order = order of pixel_values along dim 0 (see multi_view). 5) Tile pixel -> original image pixel depends…

**multi_view**

- default: config.json: dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true, force_image_size 448. IMPORTANT: the shipped modeling_internvl_chat.py never reads these four flags; tiling is done by the caller, and the model consumes pixel_values of shape (total_tiles, 3, 448, 448) plus num_patches_list. The model card (README.md / README_CN.md) contains NO preprocessing code (no dynamic_preprocess / load_image / max_num); it only calls `python example.py` from the GitHub repo, which is not in this directory. So the tiling actually used by the authors (max_num, thumbnail) is not verifiable from local files. What the modeling code fixes regardless of the preprocessing: tiles are consumed in the order of pixel_values along dim 0; all tiles of one image share one <img> ... </img> span with 256*num_tiles <IMG_CONTEXT> tokens; for several images, num_patches_list gives tiles per image and each '<image>' placeholder is replaced in order (modeling_internvl_chat.py:283-285). If '<image>' is absent from the question, chat() prepends a single '<image>\n' (line 257-258). For reference only: the standard InternVL tiling function printed in the local InternVL3_5-8B README …
- force_single_view: From the modeling code: give exactly one 448x448 tensor per image, i.e. pixel_values of shape (num_images, 3, 448, 448) and num_patches_list=[1]*num_images (single image: chat() defaults num_patches_list to [pixel_values.shape[0]] = [1], modeling_internvl_chat.py:260-261). Each image then occupies exactly 256 <IMG_CONTEXT> tokens and no thumbnail exists. How to obtain that through the authors' example.py (e.g. a max_num argument) is not stated in the local card. With the InternVL-family dynamic_preprocess (InternVL3_5-8B README), max_num=1 yields one tile and no thumbnail because of `if use_thumbnail and len(processed_images) != 1:`.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated. README.md, README_CN.md and examples/ (PNG images only) contain no grounding / bounding-box prompt, no <ref>/<box> usage and no coordinate convention. The tokenizer does define <ref> </ref> <box> </box> <quad> </quad> (added_tokens.json ids 151668-151673), inherited vocabulary only. Spatial QA prompt format as shown in the card (examples only, no formal spec): - Multi-image questions put the placeholders at the start of the question: `--question "<image><image>\nYou are standing in front of the dice pattern and observing it. Where is the desk lamp approximately located relative to you?\nOptions: A: 90 degrees counterclockwise, B: 90 degrees clockwise, C: 135 degrees counterclockwise, D: 135 degrees clockwise"` with `--image_paths examples/Q1_1.png examples/Q1_2.png` (GT: C). - Second example: `"<image><image>\nBased on these two views showing the same scene: in which direction did I move from the first view to the second view?\nA. Directly left B. Directly right C. Diagonally forward and right D. Diagonally forward and left"` (GT: D). - Multiple-choice with lettered options; no answer-format suffix is used in these two examples. For jsonl batches the card points to the…

**thinking**

- has_thinking_mode: False
- how_to_disable: Nothing to disable: the card describes no thinking mode or switch. The only related sentence is in the overview: "present a preliminary study on spatial chain-of-thought reasoning" (about the paper, not a model switch). added_tokens.json has no <think> tokens. The tokenizer chat_template and the remote chat() both insert the default InternVL system message when none is given (conversation.py:384-390, template internvl2_5).
- source: README.md (overview paragraph); added_tokens.json; tokenizer_config.json chat_template; conversation.py:384-390

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- multi_view actual preprocessing (max_num, thumbnail, tile order) used by SenseNova-SI: the card has no preprocessing code; it lives in example.py of github.com/OpenSenseNova/SenseNova-SI, not available locally. Only the config flags (dynamic 1..12 tiles + thumbnail) and the modeling-code contract are known.
- bbox.*: no grounding / bounding-box prompt or coordinate convention in README.md, README_CN.md or examples/.
- base_model variant: the front matter gives only the repo id; no sentence in the body states the exact base checkpoint or variant.
- Required transformers version: no pin in the card; only config-recorded 4.47.1 and the code assert >= 4.37.0. The uv lock of the GitHub repo is not available locally.
- vision encoder name: README does not name it; config strings (intern_vit_6b, InternViT-6B-448px-V1-5) contradict the 1024-dim / 24-layer numbers.
- Whether the remote code loads under locally installed transformers 5.x and whether the repo-prefixed auto_map forces a Hub lookup: read from source only, not tested (no weights loaded, no model-dir code imported).
- examples/examples.jsonl referenced by the README is not present in the local examples/ directory (only PNG files), so the jsonl question format could not be read.
- The example commands in this 2B card use `--model_path sensenova/SenseNova-SI-1.1-InternVL3-8B` (the 8B sibling), so no command in the card is specific to the 2B model.
- Not in MANIFEST.md: repo id / sha come only from the download log; license only from the README front matter.

### SenseNova-SI-1.2-InternVL3-8B

- **hf_repo**: sensenova/SenseNova-SI-1.2-InternVL3-8B
- **revision**: 57bfba82ed67dc27cf6607626362ee48391ecaa4
- **revision_source**: MANIFEST.md (model table row SenseNova-SI-1.2-InternVL3-8B, and the pinned `hf download ... --revision` command)
- **size_bytes**: 15904771679
- **license**: apache-2.0
- **license_source**: README.md front matter line 4 (`license: apache-2.0`); MANIFEST.md table
- **role**: spatial_finetune
- **base_model**: OpenGVLab/InternVL3-8B - from README.md front matter lines 2-3: `base_model:` / `- OpenGVLab/InternVL3-8B`. The README body does not name the base checkpoint in a sentence; the closest statements are: "we explore scaling up multimodal foundation models to cultivate spatial intelligence within the **SenseNova-SI family**, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel)." and "Currently, we build SenseNova-SI upon popular open-source foundation models to maximize comp…

**loader**

- model_class: AutoModel (config.json auto_map -> "sensenova/SenseNova-SI-1.2-InternVL3-8B--modeling_internvl_chat.InternVLChatModel"; architectures: InternVLChatModel)
- processor_class: No processor class. Tokenizer: tokenizer_config.json tokenizer_class Qwen2Tokenizer (tokenizer.json present). Image preprocessing is done by the caller; the model card contains no preprocessing code (it delegates to example.py of the GitHub repo OpenSenseNova/SenseNova-SI, which is not in this directory). preprocessor_config.json is present: feature_extractor_type CLIPFeatureExtractor, size 448, crop_size 448, do_center_crop true, resample 3, ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225]. The remote modeling code does not read it.
- trust_remote_code: True
- config_transformers_version: 4.47.1
- notes: - config.json top-level `transformers_version` is null (line 117); "4.47.1" is the value recorded inside llm_config (line 100), vision_config (line 216) and generation_config.json. - No transformers version pin appears in README.md / README_CN.md. Installation is described only as `git clone git@github.com:OpenSenseNova/SenseNova-SI.git; uv sync --extra cu124; uv sync` (the pinned versions live in that repo's lock file, not here). The remote code asserts transformers >= 4.37.0 (modeling_internvl_chat.py:46). - Usage in the card is only via `python example.py --image_paths ... --question ... --model_path sensenova/SenseNova-SI-1.2-InternVL3-8B` (example.py is not shipped here). - auto_map entries are repo-prefixed ("sensenova/SenseNova-SI-1.2-InternVL3-8B--configuration_internvl_chat.InternVLChatConfig" etc., config.json lines 7-11). In transformers 5.5.4, get_class_from_dynamic_module takes the repo id from the part before "--" (dynamic_module_utils.py:563-566) and fetches the module from that repo, so loading this local directory with AutoModel/AutoConfig may try to resolve the code from the Hub / HF cache instead of the .py files in the directory. Not tested. - Remote code suppo…

**language_model**

- model_type: qwen2
- num_layers: 28
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: All 28 layers are standard causal full self-attention with grouped-query attention (28 query heads, 4 KV heads). head_dim 128 is computed, not stated: config.json llm_config has no head_dim key; transformers Qwen2Attention uses getattr(config, "head_dim", hidden_size // num_attention_heads) = 3584 // 28 = 128 (models/qwen2/modeling_qwen2.py:195, transformers 5.5.4). No sliding window is active: llm_config has use_sliding_window: false and sliding_window: null (max_window_layers: 70 > 28 layers and inert). In transformers 5.5.4 Qwen2Config.__post_init__ sets sliding_window=None when use_sliding_window is false and builds layer_types as all "full_attention" (models/qwen2/configuration_qwen2.py:84-94). rope_scaling in llm_config: {factor: 2.0, rope_type: dynamic, type: dynamic}, rope_theta 1000000.0. llm_config._name_or_path is "./pretrained/Qwen2.5-32B-Instruct", which does not match the dimensions (hidden 3584, 28 layers) - treat it as a stale path string, not as the LLM identity.
- mlp_type: dense (architectures = Qwen2ForCausalLM; hidden_act silu, intermediate_size 18944; llm_config.moe_config is null)
- head_level_analysis: All 28 layers have the same structure (28 heads), so head-level analysis is structurally possible on every layer. Precondition: eager attention in the LLM. config.json asks for flash_attention_2 and the remote code re-forces it when use_flash_attn=True and flash_attn is importable (modeling_internvl_chat.py:55-57); load with use_flash_attn=False so the code sets 'eager'. In transformers 5.5.4 only eager_attention_forward returns the softmaxed weights (models/qwen2/modeling_qwen2.py:161-183) and config.output_attentions raises unless _attn_implementation is 'eager' (configuration_utils.py:346-351). InternVLChatModel.forward passes output_attentions to the language model and returns outputs.attentions (modeling_internvl_chat.py:137,165). Head masking is not offered by an argument in the code read; it would need hooks. Not tested (no weights were loaded). Vision encoder: InternAttention never returns attention weights (_naive_attn returns only x; FlashAttention asserts not need_weights), so ViT attention needs hooks / re-computation.

**vision_encoder**

- type: InternVisionModel from the shipped modeling_intern_vit.py. config.json vision_config.model_type is "intern_vit_6b" and vision_config._name_or_path is "OpenGVLab/InternViT-6B-448px-V1-5", but the actual dimensions are hidden_size 1024, 24 layers, 16 heads, intermediate_size 4096, norm_type layer_norm, qkv_bias true (i.e. not 6B-sized; the two strings contradict the numbers). The README does not name the vision encoder. vision_config also carries MoE-related keys (num_experts 8, use_moe false, ...) that the shipped modeling_intern_vit.py does not read.
- num_layers: 24
- hidden_size: 1024
- patch_size: 14
- resolution_strategy: Encoder input is fixed 448x448 per tile (vision_config.image_size 448, force_image_size 448) -> 32x32 = 1024 patch tokens + 1 CLS. Higher resolution is handled outside the encoder by dynamic tiling: config dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true (see multi_view).
- position_encoding: Learned absolute position embedding: nn.Parameter of shape (1, 1025, 1024) = CLS + 32x32 (modeling_intern_vit.py:149-152). At forward time the patch part goes through _get_pos_embed, which bicubic-interpolates (F.interpolate, mode='bicubic', align_corners=False) the 32x32 grid to the actual (H, W) patch grid (lines 154-173); at 448x448 input this is the identity size. No rotary / 2D relative encoding in the ViT.

**connector**

- type: pixel_shuffle (space-to-depth, downsample_ratio 0.5, ps_version v2) followed by `mlp1` = Sequential(LayerNorm(4096), Linear(4096, 3584), GELU(), Linear(3584, 3584)) (modeling_internvl_chat.py:78-83; 4096 = vit_hidden 1024 * int(1/0.5)^2; llm hidden 3584). Uses the last ViT layer (select_layer -1), CLS token dropped.
- token_merge: 2x2 patch merge by pixel_shuffle (4 ViT patch tokens -> 1 LLM token, channels concatenated 1024 -> 4096); 4x token reduction
- tokens_per_image: 256 per 448x448 tile: num_image_token = int((448 // 14) ** 2 * (0.5 ** 2)) = int(1024 * 0.25) = 256 (modeling_internvl_chat.py:52). Per image: 256 * num_tiles. In the prompt each image becomes '<img>' + '<IMG_CONTEXT>' * 256 * num_tiles + '</img>' (modeling_internvl_chat.py:284), i.e. one <img>...</img> pair per image, not per tile. Token ids (added_tokens.json): <img> 151665, </img> 151666, <IMG_CONTEXT> 151667.
- token_to_region: Derived from the shipped remote code (extract_feature + pixel_shuffle, ps_version=v2), not from a doc statement. 1) The ViT output drops the CLS token (vit_embeds[:, 1:, :]) and is reshaped to (tiles, 32, 32, 1024), row-major: dim1 = patch row (y), dim2 = patch column (x). 2) pixel_shuffle(scale_factor=0.5): view -> merges horizontally adjacent patch pairs into channels; permute(0,2,1,3); view -> merges vertically adjacent pairs; for ps_version != 'v1' a final permute(0,2,1,3) restores (row, col) order. Result (tiles, 16, 16, 4096). 3) reshape(tiles, -1, 4096) flattens row-major, so within one 448x448 tile, token k (0..255) has row i = k // 16, col j = k % 16 and covers ViT patches rows 2i..2i+1, cols 2j..2j+1, i.e. tile pixels x in [28*j, 28*j+28), y in [28*i, 28*i+28). Channel concat order inside a token: patch(2i,2j), patch(2i,2j+1), patch(2i+1,2j), patch(2i+1,2j+1). 4) Across tiles: the n-th <IMG_CONTEXT> position of an image (n counted from 0) maps to tile t = n // 256, k = n % 256, because vit_embeds.reshape(-1, C) is written into the IMG_CONTEXT positions in order. Tile order = order of pixel_values along dim 0 (see multi_view). 5) Tile pixel -> original image pixel depends…

**multi_view**

- default: config.json: dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true, force_image_size 448. IMPORTANT: the shipped modeling_internvl_chat.py never reads these four flags; tiling is done by the caller, and the model consumes pixel_values of shape (total_tiles, 3, 448, 448) plus num_patches_list. The model card (README.md / README_CN.md) contains NO preprocessing code (no dynamic_preprocess / load_image / max_num); it only calls `python example.py` from the GitHub repo, which is not in this directory. So the tiling actually used by the authors (max_num, thumbnail) is not verifiable from local files. What the modeling code fixes regardless of the preprocessing: tiles are consumed in the order of pixel_values along dim 0; all tiles of one image share one <img> ... </img> span with 256*num_tiles <IMG_CONTEXT> tokens; for several images, num_patches_list gives tiles per image and each '<image>' placeholder is replaced in order (modeling_internvl_chat.py:283-285). If '<image>' is absent from the question, chat() prepends a single '<image>\n' (line 257-258). For reference only: the standard InternVL tiling function printed in the local InternVL3_5-8B README …
- force_single_view: From the modeling code: give exactly one 448x448 tensor per image, i.e. pixel_values of shape (num_images, 3, 448, 448) and num_patches_list=[1]*num_images (single image: chat() defaults num_patches_list to [pixel_values.shape[0]] = [1], modeling_internvl_chat.py:260-261). Each image then occupies exactly 256 <IMG_CONTEXT> tokens and no thumbnail exists. How to obtain that through the authors' example.py (e.g. a max_num argument) is not stated in the local card. With the InternVL-family dynamic_preprocess (InternVL3_5-8B README), max_num=1 yields one tile and no thumbnail because of `if use_thumbnail and len(processed_images) != 1:`.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated. README.md, README_CN.md and examples/ (PNG images only) contain no grounding / bounding-box prompt, no <ref>/<box> usage and no coordinate convention. The tokenizer does define <ref> </ref> <box> </box> <quad> </quad> (added_tokens.json ids 151668-151673), inherited vocabulary only. Spatial QA prompt format as shown in the card (examples only, no formal spec): - Multi-image questions put the placeholders at the start of the question: `--question "<image><image>\nYou are standing in front of the dice pattern and observing it. Where is the desk lamp approximately located relative to you?\nOptions: A: 90 degrees counterclockwise, B: 90 degrees clockwise, C: 135 degrees counterclockwise, D: 135 degrees clockwise"` with `--image_paths examples/Q1_1.png examples/Q1_2.png` (GT: C). - Second example: `"<image><image>\nBased on these two views showing the same scene: in which direction did I move from the first view to the second view?\nA. Directly left B. Directly right C. Diagonally forward and right D. Diagonally forward and left"` (GT: D). - Multiple-choice with lettered options; no answer-format suffix is used in these two examples. For jsonl batches the card points to the…

**thinking**

- has_thinking_mode: False
- how_to_disable: Nothing to disable: the card describes no thinking mode or switch. The only related sentence is in the overview: "present a preliminary study on spatial chain-of-thought reasoning" (about the paper, not a model switch). added_tokens.json has no <think> tokens. The tokenizer chat_template and the remote chat() both insert the default InternVL system message when none is given (conversation.py:384-390, template internvl2_5).
- source: README.md (overview paragraph); added_tokens.json; tokenizer_config.json chat_template; conversation.py:384-390

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- multi_view actual preprocessing (max_num, thumbnail, tile order) used by SenseNova-SI: the card has no preprocessing code; it lives in example.py of github.com/OpenSenseNova/SenseNova-SI, not available locally. Only the config flags (dynamic 1..12 tiles + thumbnail) and the modeling-code contract are known.
- bbox.*: no grounding / bounding-box prompt or coordinate convention in README.md, README_CN.md or examples/.
- base_model variant: the front matter gives only the repo id; no sentence in the body states the exact base checkpoint or variant.
- Required transformers version: no pin in the card; only config-recorded 4.47.1 and the code assert >= 4.37.0. The uv lock of the GitHub repo is not available locally.
- vision encoder name: README does not name it; config strings (intern_vit_6b, InternViT-6B-448px-V1-5) contradict the 1024-dim / 24-layer numbers.
- Whether the remote code loads under locally installed transformers 5.x and whether the repo-prefixed auto_map forces a Hub lookup: read from source only, not tested (no weights loaded, no model-dir code imported).
- examples/examples.jsonl referenced by the README is not present in the local examples/ directory (only PNG files), so the jsonl question format could not be read.
- Differences 1.2 vs 1.3 in training data / recipe: not described in either card (only benchmark numbers and one sentence in the 1.3 card).
- README_CN.md of this 1.2 directory mentions an open-ended short-answer example (line 221) that the English README.md of 1.2 does not contain; the corresponding example images (Q3_*.png) are not in this directory's examples/.

### SenseNova-SI-1.3-InternVL3-8B

- **hf_repo**: sensenova/SenseNova-SI-1.3-InternVL3-8B
- **revision**: 2081462c2453e1331d2aa95131140b95e647b49f
- **revision_source**: MANIFEST.md (model table row SenseNova-SI-1.3-InternVL3-8B, and the pinned `hf download ... --revision` command)
- **size_bytes**: 15905828527
- **license**: apache-2.0
- **license_source**: README.md front matter line 4 (`license: apache-2.0`); MANIFEST.md table
- **role**: spatial_finetune
- **base_model**: OpenGVLab/InternVL3-8B - from README.md front matter lines 2-3: `base_model:` / `- OpenGVLab/InternVL3-8B`. The README body does not name the base checkpoint in a sentence; the closest statements are: "we explore scaling up multimodal foundation models to cultivate spatial intelligence within the **SenseNova-SI family**, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel)." and "Currently, we build SenseNova-SI upon popular open-source foundation models to maximize comp…

**loader**

- model_class: AutoModel (config.json auto_map -> "sensenova/SenseNova-SI-1.3-InternVL3-8B--modeling_internvl_chat.InternVLChatModel"; architectures: InternVLChatModel)
- processor_class: No processor class. Tokenizer: tokenizer_config.json tokenizer_class Qwen2Tokenizer (tokenizer.json present). Image preprocessing is done by the caller; the model card contains no preprocessing code (it delegates to example.py of the GitHub repo OpenSenseNova/SenseNova-SI, which is not in this directory). preprocessor_config.json is present: feature_extractor_type CLIPFeatureExtractor, size 448, crop_size 448, do_center_crop true, resample 3, ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225]. The remote modeling code does not read it.
- trust_remote_code: True
- config_transformers_version: 4.47.1
- notes: - config.json top-level `transformers_version` is null (line 117); "4.47.1" is the value recorded inside llm_config (line 100), vision_config (line 216) and generation_config.json. - No transformers version pin appears in README.md / README_CN.md. Installation is described only as `git clone git@github.com:OpenSenseNova/SenseNova-SI.git; uv sync --extra cu124; uv sync` (the pinned versions live in that repo's lock file, not here). The remote code asserts transformers >= 4.37.0 (modeling_internvl_chat.py:46). - Usage in the card is only via `python example.py --image_paths ... --question ... --model_path sensenova/SenseNova-SI-1.3-InternVL3-8B` (example.py is not shipped here). - auto_map entries are repo-prefixed ("sensenova/SenseNova-SI-1.3-InternVL3-8B--configuration_internvl_chat.InternVLChatConfig" etc., config.json lines 7-11). In transformers 5.5.4, get_class_from_dynamic_module takes the repo id from the part before "--" (dynamic_module_utils.py:563-566) and fetches the module from that repo, so loading this local directory with AutoModel/AutoConfig may try to resolve the code from the Hub / HF cache instead of the .py files in the directory. Not tested. - Remote code suppo…

**language_model**

- model_type: qwen2
- num_layers: 28
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: All 28 layers are standard causal full self-attention with grouped-query attention (28 query heads, 4 KV heads). head_dim 128 is computed, not stated: config.json llm_config has no head_dim key; transformers Qwen2Attention uses getattr(config, "head_dim", hidden_size // num_attention_heads) = 3584 // 28 = 128 (models/qwen2/modeling_qwen2.py:195, transformers 5.5.4). No sliding window is active: llm_config has use_sliding_window: false and sliding_window: null (max_window_layers: 70 > 28 layers and inert). In transformers 5.5.4 Qwen2Config.__post_init__ sets sliding_window=None when use_sliding_window is false and builds layer_types as all "full_attention" (models/qwen2/configuration_qwen2.py:84-94). rope_scaling in llm_config: {factor: 2.0, rope_type: dynamic, type: dynamic}, rope_theta 1000000.0. llm_config._name_or_path is "./pretrained/Qwen2.5-32B-Instruct", which does not match the dimensions (hidden 3584, 28 layers) - treat it as a stale path string, not as the LLM identity.
- mlp_type: dense (architectures = Qwen2ForCausalLM; hidden_act silu, intermediate_size 18944; llm_config.moe_config is null)
- head_level_analysis: All 28 layers have the same structure (28 heads), so head-level analysis is structurally possible on every layer. Precondition: eager attention in the LLM. config.json asks for flash_attention_2 and the remote code re-forces it when use_flash_attn=True and flash_attn is importable (modeling_internvl_chat.py:55-57); load with use_flash_attn=False so the code sets 'eager'. In transformers 5.5.4 only eager_attention_forward returns the softmaxed weights (models/qwen2/modeling_qwen2.py:161-183) and config.output_attentions raises unless _attn_implementation is 'eager' (configuration_utils.py:346-351). InternVLChatModel.forward passes output_attentions to the language model and returns outputs.attentions (modeling_internvl_chat.py:137,165). Head masking is not offered by an argument in the code read; it would need hooks. Not tested (no weights were loaded). Vision encoder: InternAttention never returns attention weights (_naive_attn returns only x; FlashAttention asserts not need_weights), so ViT attention needs hooks / re-computation.

**vision_encoder**

- type: InternVisionModel from the shipped modeling_intern_vit.py. config.json vision_config.model_type is "intern_vit_6b" and vision_config._name_or_path is "OpenGVLab/InternViT-6B-448px-V1-5", but the actual dimensions are hidden_size 1024, 24 layers, 16 heads, intermediate_size 4096, norm_type layer_norm, qkv_bias true (i.e. not 6B-sized; the two strings contradict the numbers). The README does not name the vision encoder. vision_config also carries MoE-related keys (num_experts 8, use_moe false, ...) that the shipped modeling_intern_vit.py does not read.
- num_layers: 24
- hidden_size: 1024
- patch_size: 14
- resolution_strategy: Encoder input is fixed 448x448 per tile (vision_config.image_size 448, force_image_size 448) -> 32x32 = 1024 patch tokens + 1 CLS. Higher resolution is handled outside the encoder by dynamic tiling: config dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true (see multi_view).
- position_encoding: Learned absolute position embedding: nn.Parameter of shape (1, 1025, 1024) = CLS + 32x32 (modeling_intern_vit.py:149-152). At forward time the patch part goes through _get_pos_embed, which bicubic-interpolates (F.interpolate, mode='bicubic', align_corners=False) the 32x32 grid to the actual (H, W) patch grid (lines 154-173); at 448x448 input this is the identity size. No rotary / 2D relative encoding in the ViT.

**connector**

- type: pixel_shuffle (space-to-depth, downsample_ratio 0.5, ps_version v2) followed by `mlp1` = Sequential(LayerNorm(4096), Linear(4096, 3584), GELU(), Linear(3584, 3584)) (modeling_internvl_chat.py:78-83; 4096 = vit_hidden 1024 * int(1/0.5)^2; llm hidden 3584). Uses the last ViT layer (select_layer -1), CLS token dropped.
- token_merge: 2x2 patch merge by pixel_shuffle (4 ViT patch tokens -> 1 LLM token, channels concatenated 1024 -> 4096); 4x token reduction
- tokens_per_image: 256 per 448x448 tile: num_image_token = int((448 // 14) ** 2 * (0.5 ** 2)) = int(1024 * 0.25) = 256 (modeling_internvl_chat.py:52). Per image: 256 * num_tiles. In the prompt each image becomes '<img>' + '<IMG_CONTEXT>' * 256 * num_tiles + '</img>' (modeling_internvl_chat.py:284), i.e. one <img>...</img> pair per image, not per tile. Token ids (added_tokens.json): <img> 151665, </img> 151666, <IMG_CONTEXT> 151667.
- token_to_region: Derived from the shipped remote code (extract_feature + pixel_shuffle, ps_version=v2), not from a doc statement. 1) The ViT output drops the CLS token (vit_embeds[:, 1:, :]) and is reshaped to (tiles, 32, 32, 1024), row-major: dim1 = patch row (y), dim2 = patch column (x). 2) pixel_shuffle(scale_factor=0.5): view -> merges horizontally adjacent patch pairs into channels; permute(0,2,1,3); view -> merges vertically adjacent pairs; for ps_version != 'v1' a final permute(0,2,1,3) restores (row, col) order. Result (tiles, 16, 16, 4096). 3) reshape(tiles, -1, 4096) flattens row-major, so within one 448x448 tile, token k (0..255) has row i = k // 16, col j = k % 16 and covers ViT patches rows 2i..2i+1, cols 2j..2j+1, i.e. tile pixels x in [28*j, 28*j+28), y in [28*i, 28*i+28). Channel concat order inside a token: patch(2i,2j), patch(2i,2j+1), patch(2i+1,2j), patch(2i+1,2j+1). 4) Across tiles: the n-th <IMG_CONTEXT> position of an image (n counted from 0) maps to tile t = n // 256, k = n % 256, because vit_embeds.reshape(-1, C) is written into the IMG_CONTEXT positions in order. Tile order = order of pixel_values along dim 0 (see multi_view). 5) Tile pixel -> original image pixel depends…

**multi_view**

- default: config.json: dynamic_image_size true, min_dynamic_patch 1, max_dynamic_patch 12, use_thumbnail true, force_image_size 448. IMPORTANT: the shipped modeling_internvl_chat.py never reads these four flags; tiling is done by the caller, and the model consumes pixel_values of shape (total_tiles, 3, 448, 448) plus num_patches_list. The model card (README.md / README_CN.md) contains NO preprocessing code (no dynamic_preprocess / load_image / max_num); it only calls `python example.py` from the GitHub repo, which is not in this directory. So the tiling actually used by the authors (max_num, thumbnail) is not verifiable from local files. What the modeling code fixes regardless of the preprocessing: tiles are consumed in the order of pixel_values along dim 0; all tiles of one image share one <img> ... </img> span with 256*num_tiles <IMG_CONTEXT> tokens; for several images, num_patches_list gives tiles per image and each '<image>' placeholder is replaced in order (modeling_internvl_chat.py:283-285). If '<image>' is absent from the question, chat() prepends a single '<image>\n' (line 257-258). For reference only: the standard InternVL tiling function printed in the local InternVL3_5-8B README …
- force_single_view: From the modeling code: give exactly one 448x448 tensor per image, i.e. pixel_values of shape (num_images, 3, 448, 448) and num_patches_list=[1]*num_images (single image: chat() defaults num_patches_list to [pixel_values.shape[0]] = [1], modeling_internvl_chat.py:260-261). Each image then occupies exactly 256 <IMG_CONTEXT> tokens and no thumbnail exists. How to obtain that through the authors' example.py (e.g. a max_num argument) is not stated in the local card. With the InternVL-family dynamic_preprocess (InternVL3_5-8B README), max_num=1 yields one tile and no thumbnail because of `if use_thumbnail and len(processed_images) != 1:`.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: Not stated. README.md, README_CN.md and examples/ (PNG images only) contain no grounding / bounding-box prompt, no <ref>/<box> usage and no coordinate convention. The tokenizer does define <ref> </ref> <box> </box> <quad> </quad> (added_tokens.json ids 151668-151673), inherited vocabulary only. Spatial QA prompt format as shown in the card (examples only, no formal spec): - Example 1 (single image, SITE-Bench): `--question "Consider the real-world 3D locations of the objects. Which is closer to the sink, the toilet paper or the towel?\nOptions: \nA. toilet paper\nB. towel\nGive me the answer letter directly. The best answer is:"` (GT: A). - Example 2 (two images, MMSI-Bench): `--question "If the landscape painting is on the east side of the bedroom, where is the window located in the bedroom?\nOptions: A. North side, B. South side, C. West side, D. East side\nAnswer with the option's letter from the given choices directly. Enclose the option's letter within ``."` (GT: C). - Example 3 (three images, open-ended short answer): `--question "The robot is making tea. What is the order in which the pictures were taken?"` (GT: Second, first, third). - Unlike the 1.1 / 1.2 cards, the 1.3 e…

**thinking**

- has_thinking_mode: False
- how_to_disable: Nothing to disable: the card describes no thinking mode or switch. The only related sentence is in the overview: "present a preliminary study on spatial chain-of-thought reasoning" (about the paper, not a model switch). added_tokens.json has no <think> tokens. The tokenizer chat_template and the remote chat() both insert the default InternVL system message when none is given (conversation.py:384-390, template internvl2_5).
- source: README.md (overview paragraph); added_tokens.json; tokenizer_config.json chat_template; conversation.py:384-390

**查不到 / 未确认**

- env: left null by instruction (decided in stage B).
- multi_view actual preprocessing (max_num, thumbnail, tile order) used by SenseNova-SI: the card has no preprocessing code; it lives in example.py of github.com/OpenSenseNova/SenseNova-SI, not available locally. Only the config flags (dynamic 1..12 tiles + thumbnail) and the modeling-code contract are known.
- bbox.*: no grounding / bounding-box prompt or coordinate convention in README.md, README_CN.md or examples/.
- base_model variant: the front matter gives only the repo id; no sentence in the body states the exact base checkpoint or variant.
- Required transformers version: no pin in the card; only config-recorded 4.47.1 and the code assert >= 4.37.0. The uv lock of the GitHub repo is not available locally.
- vision encoder name: README does not name it; config strings (intern_vit_6b, InternViT-6B-448px-V1-5) contradict the 1024-dim / 24-layer numbers.
- Whether the remote code loads under locally installed transformers 5.x and whether the repo-prefixed auto_map forces a Hub lookup: read from source only, not tested (no weights loaded, no model-dir code imported).
- examples/examples.jsonl referenced by the README is not present in the local examples/ directory (only PNG files), so the jsonl question format could not be read.
- Differences 1.2 vs 1.3 in training data / recipe: not described; the card only says 1.3 'improves open-ended spatial question-answering capabilities compared to previous versions' and gives scores.

### VST-7B-RL

- **hf_repo**: rayruiyang/VST-7B-RL
- **revision**: b18ec54b43e1d0b8a171cf41d9f221fbedd803ae
- **revision_source**: MANIFEST.md table row (line 15) and reproduce command (line 39)
- **size_bytes**: 16600362430
- **license**: apache-2.0
- **license_source**: README.md front matter line 2 (MANIFEST.md also lists apache-2.0)
- **role**: spatial_finetune
- **base_model**: Qwen2.5-VL 7B (exact hub repo not stated). The model card does not name the base; evidence is config.json _name_or_path ".../0916_qwen2_5_vl_7b_grpo_dapo_mix_final/global_step_750/actor/huggingface", model_type qwen2_5_vl, and the README snippet loading it with Qwen2_5_VLForConditionalGeneration. "RL" per the card: "Progressive Training Pipeline: Start with supervised fine-tuning to build foundational spatial knowledge, then reinforce spatial reasoning abilities via RL" (README:39); VST-R is "135K curated samples that teach models to reason in space, including step-by-step reasoning and rule-…

**loader**

- model_class: Qwen2_5_VLForConditionalGeneration
- processor_class: Qwen2_5_VLProcessor
- trust_remote_code: False
- config_transformers_version: 4.49.0
- notes: Native transformers class; README quickstart: "pip install transformers" (no version pin) and "pip install qwen-vl-utils"; loads with torch_dtype=torch.bfloat16 and builds the processor as AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28), i.e. the card's example overrides the shipped preprocessor limits. preprocessor_config.json names image_processor_type Qwen2VLImageProcessorFast. config.json vision_config is sparse (only hidden_size=1280, in_chans, spatial_patch_size, tokens_per_second=2, torch_dtype); every other vision field falls back to Qwen2_5_VLVisionConfig class defaults: depth=32, intermediate_size=3420, num_heads=16, patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=112, out_hidden_size=3584, fullatt_block_indexes=(7,15,23,31), hidden_act=silu, in_channels=3. The weight index agrees with depth 32 (visual.blocks.0..31). MANIFEST.md records a config + processor smoke test under transformers 5.5.4 (no weights loaded).

**language_model**

- model_type: qwen2_5_vl_text
- num_layers: 28
- hidden_size: 3584
- num_attention_heads: 28
- num_key_value_heads: 4
- head_dim: 128
- layer_mixers（逐层，0 起）: `F F F F F F F F F F F F F F F F F F F F F F F F F F F F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: All 28 layers are plain causal self-attention. config.json has sliding_window=32768 and max_window_layers=28 but use_sliding_window=false; Qwen2_5_VLTextConfig.__post_init__ sets sliding_window=None when use_sliding_window is false and then builds layer_types as "sliding_attention" only if sliding_window is not None and i >= max_window_layers, so every layer is "full_attention" (and even with the switch on, i >= 28 never holds for 28 layers). Grouped-query attention: 28 query heads, 4 key/value heads (each kv head shared by 7 query heads). Position encoding in the LM is multimodal RoPE with mrope_section [16, 24, 24], rope_theta 1e6.
- mlp_type: dense
- head_level_analysis: All 28 layers have the same standard attention structure (q_proj -> 28 heads x 128 dims, o_proj takes the concatenated 28 heads), so head-level analysis / attention masking applies to every layer. Note k/v are shared within groups of 7 query heads (4 kv heads). Whether per-head attention weights are returned under the sdpa/flash backends was not checked here.

**vision_encoder**

- type: Qwen2.5-VL ViT (Qwen2_5_VisionTransformerPretrainedModel) with window attention
- num_layers: 32
- hidden_size: 1280
- patch_size: 14
- resolution_strategy: dynamic (native aspect ratio). Image processor smart_resize: round H and W to multiples of 28 (= patch 14 x merge 2); if H*W > max_pixels scale down (floor to multiple of 28), if H*W < min_pixels scale up (ceil to multiple of 28); aspect ratio > 200 raises. Values shipped in this model dir's preprocessor_config.json: min_pixels=3136 (56x56 -> 4 tokens), max_pixels=12845056 (= 16384 x 28 x 28 -> up to 16384 tokens). Resample bicubic, CLIP mean/std. The model card's example instead uses min_pixels=256*28*28 (200704) and max_pixels=1280*28*28 (1003520), i.e. 256 to 1280 tokens; the card also routes images through qwen_vl_utils.process_vision_info, whose own resizing was not read.
- position_encoding: 2D rotary position embedding (RoPE) on (row, col) patch indices inside every vision block (rot_pos_emb; no learned absolute position table). Window attention: 28 of 32 blocks attend only inside 112x112 px windows (= 8x8 patches = 4x4 merged cells); blocks 7, 15, 23, 31 (fullatt_block_indexes) attend over the whole image. 16 heads, temporal_patch_size 2 (a still image is duplicated to 2 frames and embedded by a Conv3d).

**connector**

- type: Qwen2_5_VLPatchMerger: RMSNorm, concatenate each 2x2 group of patch features (4 x 1280 = 5120), then MLP Linear(5120,5120) -> GELU -> Linear(5120, 3584)
- token_merge: 2x2 spatial merge (4 patches -> 1 token), spatial_merge_size 2
- tokens_per_image: (H'/28) * (W'/28), where (H', W') is the smart_resize output (both multiples of 28). Processor: image_grid_thw.prod() // merge_size**2 with grid_t = 1 for a still image. The tokens are <|image_pad|> placeholders wrapped as <|vision_start|> ... <|vision_end|> by the chat template. With shipped limits: 4 to 16384 tokens per image.
- token_to_region: The tokens that reach the LM ARE in raster (row-major) order of merged 28x28 px cells. Details: the image processor orders patches as (merged_row, merged_col, 2x2 inside the cell), the vision tower then permutes merged cells into window order (window_index) for window attention, the merger fuses each consecutive group of 4 patches, and finally merged_hidden_states[argsort(window_index)] restores the original order before the features are masked_scatter-ed into the <|image_pad|> positions. Rule: with gw = W'/28, the k-th image token (0-based) is row r = k // gw, col c = k % gw and covers resized-image pixels x in [28c, 28c+28), y in [28r, 28r+28); in original-image pixels multiply x by W/W' and y by H/H'. CAVEAT for vision-tower internals: the pre-merger hidden states (last_hidden_state / per-block outputs) are left in WINDOW order, not raster order; only the merged pooler_output is un-permuted, so apply argsort(window_index) yourself when reading intermediate ViT activations.

**multi_view**

- default: Single view. No tiling, no thumbnail, no newline tokens; one native-resolution grid per image.
- force_single_view: Nothing to disable. To fix the token count, pass min_pixels / max_pixels (processor or image-processor kwargs) or pre-resize the image to multiples of 28.

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md searched for grounding / bounding box / bbox / point / coordinate: no match. The only example prompt is a spatial yes/no question (README:109).

**thinking**

- has_thinking_mode: True
- how_to_disable: Thinking is prompt-driven, not a chat-template flag: the card prepends a system message THINK_SYSTEM_PROMPT = "You are a helpful assistant. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here." when its Python variable enable_thinking is True (README:80-86, 114-115). To disable, do not insert that system message (the chat template then uses the default "You are a helpful assistant."). The chat template itself has no enable_thinking argument. Whether the RL model still emits <think> without the prompt is not stated.
- source: README.md:80-86 and 114-115; chat_template.json:2

**查不到 / 未确认**

- base_model exact repo id: the card never names it; 'qwen2_5_vl_7b' appears only in config.json _name_or_path, so whether it is Qwen/Qwen2.5-VL-7B-Instruct specifically is not confirmed by any local file.
- bbox.*: no grounding prompt, output format or coordinate convention in the model card.
- thinking without the system prompt: the card does not say whether reasoning is still produced; answer format after </think> is not specified beyond the prompt text.
- qwen_vl_utils.process_vision_info resizing behaviour (used in the card's example) was not read; only the transformers image processor path is documented here.
- config.json vision_config uses legacy key names (in_chans, spatial_patch_size) that are not fields of Qwen2_5_VLVisionConfig in transformers 5.5.4; how they are handled at load time was not executed here (MANIFEST.md reports the config loads).

### gemma-4-31B-it

- **hf_repo**: google/gemma-4-31B-it
- **revision**: 842da3794eaa0b77d5f08bae87a17459d91ff475
- **revision_source**: MANIFEST.md:12（表格）与 :36（hf download --revision）；与 .cache/huggingface/download/config.json.metadata 第 1 行一致
- **size_bytes**: 62578686256
- **license**: apache-2.0
- **license_source**: README.md:3 front matter `license: apache-2.0`（:4 license_link 指向 ai.google.dev/gemma/docs/gemma_4_license）；MANIFEST.md:12 同为 apache-2.0
- **role**: general
- **base_model**: —

**loader**

- model_class: Gemma4ForConditionalGeneration
- processor_class: Gemma4Processor
- trust_remote_code: False
- config_transformers_version: 5.5.0.dev0
- notes: README 只写 "You can use all Gemma 4 models with the latest version of Transformers" 与 `pip install -U transformers torch accelerate`（图像另加 torchvision），未给最低版本号。 README 示例：AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")。 tokenizer_class=GemmaTokenizer。processor 默认返回 mm_token_type_ids（processing_gemma4.py:41）， 模型靠它构造图像块双向 mask（见 mixer_details），自己拼输入时不要丢掉。 masking_utils.py:1173-1175：使用 or_mask_function（即图像双向 mask）要求 torch>=2.6。 generation_config 默认 do_sample=true, temperature=1.0, top_k=64, top_p=0.95。

**language_model**

- model_type: gemma4_text
- num_layers: 60
- hidden_size: 5376
- num_attention_heads: 32
- num_key_value_heads: 16
- head_dim: 256
- layer_mixers（逐层，0 起）: `S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F`  （F=标准自注意力，L=线性注意力/Gated DeltaNet，S=滑动窗口自注意力）
- mixer_details: 模式：每 6 层一组 = 5 层 sliding_attention + 1 层 full_attention，共 50 层 sliding、10 层 full （full 层下标 5,11,17,23,29,35,41,47,53,59；config.json:37 起的 layer_types，已用 python 核对长度 60）。 【两类层都是标准 softmax 自注意力】同一个 Gemma4TextAttention 类（modeling_gemma4.py:1126）， 差别只在 mask 与投影形状；eager 路径 eager_attention_forward（:768-799）为 QK^T*scaling → 加 additive mask → softmax → 乘 V，返回 attn_weights。scaling=1.0（:1143）， q/k 先过 RMSNorm（q_norm/k_norm :1165,1169）再加 RoPE；文本注意力未传 softcap。 【sliding 层】sliding_window=1024（config.json:120）。mask = 因果 AND 窗口： kv_idx <= q_idx 且 kv_idx > q_idx - sliding_window（masking_utils.py:78, :90-99, :114-118）， 即每个 query 只能看到自己和之前的 1023 个 token；更远的 token 被 mask（注意力权重为 0，不是近似）。 q 头 32，head_dim=256，KV 头 num_key_value_heads=16（每 2 个 q 头共享一组 KV，repeat_kv :782-783）， 有独立 v_proj。RoPE: theta=10000, rope_type default。 【full 层】纯因果全局 mask（create_causal_mask，:2063）。q 头 32，head_dim=global_head_dim=512（:1137）， KV 头 num_global_key_value_heads=4（:1139-1142，每 8 个 q 头共享一组 KV）。 attention_k_eq_v=true 只作用于 full 层（use_alternative_attention = attention_k_eq_v and not is_sliding，:1138）： 不建 v_proj（:1175-1179），value_states 直接取 k_proj 的输出（:1214），之后 K 走 k_norm+RoPE、V 走 v_norm（无 scale，:1170,:1216-1221）。 RoPE: theta=1e6,…
- mlp_type: dense（enable_moe_block=false, num_experts=null；Gemma4TextMLP 为 gate/up/down 门控 MLP，intermediate_size=21504，use_double_wide_mlp=false）
- head_level_analysis: 60 层都可做 head 级分析与注意力屏蔽（都是 softmax 注意力，o_proj 输入是 32 个头按 head_dim 拼接，:1181-1183,:1244-1245）， 但要注意：(1) sliding 层每头 256 维、full 层每头 512 维，o_proj 输入宽度分别为 8192 / 16384； (2) KV 是分组共享的：sliding 层 16 组（2 头/组），full 层 4 组（8 头/组）且 V 与 K 同源（k_eq_v）， 同组的头只有 Q 不同，对 K/V 做干预会同时影响整组； (3) sliding 层里距离 query >= 1024 的 token 权重恒为 0。图像最多 280 个 token， 若图像位于答案位置之前 1024 token 以外（长提示词/多图/开启 thinking 后的长思考）， 这些图像 token 在 50 个 sliding 层不可直接到达，只能在 10 个 full 层被直接读取，或经窗口内的中间 token 间接传递； (4) sliding 层的图像块内双向 mask 使“屏蔽图像 token”的实验需同时考虑图像 token 之间的双向边； (5) attn_weights 由 eager_attention_forward 返回；非 eager 实现走 ALL_ATTENTION_FUNCTIONS（:1228-1230）， 是否返回权重未在此核实，做注意力分析建议 attn_implementation="eager"。

**vision_encoder**

- type: Gemma4VisionModel (model_type gemma4_vision)；README 称 31B 的 vision encoder 约 550M 参数
- num_layers: 27
- hidden_size: 1152
- patch_size: 16
- resolution_strategy: dynamic（可变分辨率 + 可变长宽比，单视图）。图像处理器按 token 预算做保持长宽比的缩放 （image_processing_gemma4.py:162-181；get_aspect_ratio_preserving_size，image_processing_pil_gemma4.py:36-86）： max_patches = max_soft_tokens * pooling_kernel_size^2 = 280*9 = 2520 个 16px patch（目标像素 645120）； factor = sqrt(target_px / (H*W))，H、W 各自乘 factor 后向下取整到 48（=pooling_kernel_size*patch_size）的倍数； 没有“只缩不放”的条件，小图也会被放大到预算附近。再切成 16px patch，并 pad 到 max_patches（位置 id 填 -1）。 max_soft_tokens 默认 280（processor_config.json:42），只允许 {70,140,280,560,1120} （image_processing_pil_gemma4.py:33；README:398 同）。do_normalize=false，像素 rescale 到 [0,1] 后在模型内做 2*(x-0.5)（modeling_gemma4.py:567）。
- position_encoding: 两种叠加：(1) 可学习的分解式 2D 绝对位置表 position_embedding_table，形状 [2, position_embedding_size=10240, hidden]， 按 patch 的 (x, y) 坐标分别查表后相加，加到 patch embedding 上，padding patch 置 0（modeling_gemma4.py:548-570）； (2) 每层注意力内的 2D RoPE：对 x、y 两个轴各用 head_dim 的一半独立计算频率（rope_theta=100，:686-700, :704-724）， apply_multidimensional_rope（:802-855）。编码器为双向全局注意力（create_bidirectional_mask，:993），padding patch 被 mask。

**connector**

- type: Gemma4VisionPooler（按位置的 3x3 平均池化 + 乘 sqrt(hidden)）→ 可选 standardize（减 std_bias 乘 std_scale）→ embed_vision（Gemma4MultimodalEmbedder 投影到 LLM 维度）
- token_merge: 3x3 平均池化，9 个 16px patch → 1 个 soft token（pooling_kernel_size=3；modeling_gemma4.py:581-606）
- tokens_per_image: 可变，不是固定 280。n = (H'/48) * (W'/48)，H'、W' 为缩放后的高宽（48 的倍数），n <= max_soft_tokens（默认 280）。 processor 用每张图实际的 num_soft_tokens 展开占位符：<|image> + <|image|>*n + <image|>（processing_gemma4.py:150； image_processing_gemma4.py:191），pad 出来的 token 在模型里被丢弃（modeling_gemma4.py:1941-1942）。 按代码推算的例子（未实测）：正方形图 → 768x768 → 16x16 = 256 个 token。 config.vision_soft_tokens_per_image=280 / image_seq_length=280 是上限/默认预算。
- token_to_region: 可行，规则确定。patch 按行优先排列（meshgrid indexing="xy"，位置为 (x, y)，image_processing_gemma4.py:195-201）； 池化索引 = floor(x/3) + (W_patches/3) * floor(y/3)（modeling_gemma4.py:600-602），所以 soft token 也是行优先： 第 i 个 token ↔ 池化网格 row = i // (W'/48), col = i % (W'/48)，对应缩放后图像上的 48x48 像素格 [col*48, (col+1)*48) x [row*48, (row+1)*48)；再按 (原宽/W', 原高/H') 映回原图（长宽比因取整到 48 的倍数只近似保持）。 注意每个 token 是 27 层全局双向注意力之后再池化的，内容上会混入全图信息。

**multi_view**

- default: 否。Gemma4 processor / image processor 中没有 pan&scan、切片或缩略图逻辑（grep pan_and_scan|crop 无结果），每张图恒为单视图，token 数随长宽比变化。
- force_single_view: 无需设置（本来就是单视图）。要固定 token 数与网格：保持 max_soft_tokens=280（或经 processor 的 images_kwargs 传 max_soft_tokens ∈ {70,140,280,560,1120}），并把输入图预先缩放到固定长宽比/尺寸（例如 48 的倍数），使 (H'/48, W'/48) 固定。

**bbox**

- official_prompt: —
- output_format: —
- coord_range: —
- coord_order: —
- source: README.md:119 只在能力列表里提到 "Object detection ... and pointing"，全文没有给出检测/框的提示词、输出格式或坐标约定；chat_template.jinja 中也没有相关内容。

**thinking**

- has_thinking_mode: True
- how_to_disable: apply_chat_template(..., enable_thinking=False)，这也是模板默认值（chat_template.jinja:186 `enable_thinking | default(false)`）。 关闭时模板不在 system turn 写 <|think|>，并在 generation prompt 的 `<|turn>model\n` 后预填空思考块 `<|channel>thought\n<channel|>`（:383-385）。开启时在第一个 system turn 顶部注入 `<|think|>\n`（:193-194）。 README.md:374 原文："Thinking is enabled by including the `<|think|>` token at the start of the system prompt. To disable thinking, remove the token." README.md:377："For all models except for the E2B and E4B variants, if thinking is disabled, the model will still generate the tags but with an empty thought block"。 README 的纯文本示例显式传 enable_thinking=False（:165），图像示例没有传该参数（即取模板默认 false）。
- source: README.md:165,177,370-378；chat_template.jinja:186,193-194,383-385

**查不到 / 未确认**

- bbox.official_prompt / output_format / coord_range / coord_order：README 与 chat_template 均未给出检测或框的提示词与坐标约定（只在能力列表提到 object detection / pointing）。
- loader 最低 transformers 版本：README 只说 latest version，未给版本号；config 记录 5.5.0.dev0，本机 5.5.4 自带 gemma4 实现，但未加载权重验证。
- 非 eager 注意力实现（sdpa/flash）是否返回 attn_weights：未核实。
- 图像块双向 mask 在增量解码（带 KV cache 的后续步）中的行为：仅读了 create_causal_mask_mapping 的 is_first_iteration 条件，未实测。
- tokens_per_image 的具体数值例子是按源码公式手算，未用真实图像跑 processor 验证。
- _logs/hf_download_*.out 中未找到 gemma 的 repo id 或 sha（grep 无结果），sha 仅来自 MANIFEST.md 与 .cache 元数据。

