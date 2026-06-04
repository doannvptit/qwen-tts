File mới: src/components/mel_discriminator.py
- MelSubDiscriminator (sub-D với 5 conv + spectral norm) — y nguyên bản của bạn
- MultiScaleMelDiscriminator (3 sub-D ở 3 scale qua AvgPool) — y nguyên bản
- 3 hàm loss LSGAN: discriminator_loss, generator_adversarial_loss, feature_matching_loss
src/model.py:
- LlmSpokenModelConfig thêm discriminator_warmup_steps=10000, discriminator_loss_weight=1.0, feature_matching_loss_weight=1.0 (đọc từ yaml).
- self.discriminator = MultiScaleMelDiscriminator(n_mels=mel_bins) chạy ở float32.
- self.disc_step là buffer để checkpoint có thể resume đúng warmup.
- is_discriminator_active(): True khi disc_step >= warmup.
- forward(): khi active, chạy D trên mel_post (có grad → cập nhật G) và audio_mels (no_grad → lấy fmap thật) rồi thêm adv_loss, feat_match_loss vào output. Trước warmup, cả hai loss bằng 0.
- discriminator_step(mel_post, audio_mels): chạy D trên cả real/fake (có grad → cập nhật D), trả loss hoặc None nếu chưa warmup.
- optimizer_param_groups() (G) và discriminator_param_groups() (D) tách riêng; generator_parameters() phục vụ grad clip.
- save_pretrained / from_pretrained lưu discriminator.safetensors cùng talker.safetensors.
train.py:
- TrainingConfig thêm discriminator_learning_rate=2e-4, discriminator_warmup_steps=10000, discriminator_loss_weight, feature_matching_loss_weight.
- Tạo disc_optimizer AdamW riêng cho D, scheduler riêng (warmup = disc_warmup_steps - lr_warmup_steps).
- Mỗi microbatch: tính d_loss = model.discriminator_step(...), backward D (chia cho grad_accum), backward G (chia cho grad_accum) trong cùng no_sync() (DDP) khi chưa đến step update.
- Khi update: clip grad riêng cho G và D, step cả hai, zero_grad cả hai, tăng global_step rồi raw_model.disc_step.fill_(global_step) (tự kích hoạt sau 10000 step).
- save_checkpoint lưu thêm disc_optimizer / disc_scheduler state.
- Logging: loss/adv, loss/feat_match, loss/d, train/disc_active, train/disc_lr.
Yaml cập nhật: configs/Qwen3-0.6B-Instruct-freeze.yaml (3 trường discriminator cho model), trainings/Qwen3-0.6B-Instruct-freeze.yaml (4 trường cho training).
Lưu ý: hai lần forward qua D (một cho adv ở forward(), một cho D step) sẽ áp dụng dropout khác nhau; nếu muốn D chỉ học từ d_loss thuần, có thể wrap self.discriminator(mel_post_f) trong torch.no_grad() rồi tính adv qua feature matching + một score detached. Theo cách HiFi-GAN gốc thì giữ nguyên như trên.