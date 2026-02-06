import os
import torch
import torchaudio
import copy

from CCS_metrics import compute_cer, compute_wer
from cosine import WarmupCosineScheduler, WarmupCosineRestartScheduler
from datamodule.transforms import TextTransform, TextTransform_CCS
from datamodule.transforms_english import TextTransform_English

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer


class EMA:
    """
    Exponential Moving Average for model weights.

    Maintains a shadow copy of model weights that is updated with exponential
    moving average of training weights. This often leads to better generalization.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        """Register shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow weights with EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


class ModelModule_CCS(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # Skip saving hyperparameters for complex config objects
        # to avoid serialization issues with nested structures
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone

        # Determine tokenizer based on dataset
        dataset_name = getattr(cfg.data.dataset, '_name_', '') if hasattr(cfg.data, 'dataset') else ''
        # Also check root_dir for English dataset detection
        root_dir = getattr(cfg.data.dataset, 'root_dir', '') if hasattr(cfg.data, 'dataset') else ''
        use_english = 'english' in str(dataset_name).lower() or 'lrs2' in str(dataset_name).lower() or 'mvlrs' in str(root_dir).lower()

        if use_english:
            print(f"  ✓ Using English character-level tokenizer (30 tokens)")
            self.text_transform = TextTransform_English()
        else:
            # Check if we should use BPE (for pretrained models) or CCS units (for training)
            use_bpe = getattr(cfg, "use_bpe_tokenizer", False)

            if use_bpe:
                print(f"  ✓ Using BPE tokenizer (5049 tokens) for pretrained model")
                self.text_transform = TextTransform(target_vocab_size=5049)
            else:
                # Fallback to CCS units for training
                units_file = "data/multilingual_units.txt"
                if os.path.exists(units_file):
                    print(f"  ✓ Loading multilingual units from {units_file}")
                    self.text_transform = TextTransform_CCS(units_file=units_file)
                else:
                    print(f"  ✓ Using CCS tokenizer (44 tokens)")
                    self.text_transform = TextTransform_CCS()

        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # Idea 1: Semantic Alignment Projections
        # Project visual and text features to same dimension for contrastive learning
        self.align_proj_visual = torch.nn.Linear(self.backbone_args.adim, 256)
        self.align_proj_text = torch.nn.Linear(self.backbone_args.ddim, 256)
        self.align_temperature = 0.07

        # EMA (Exponential Moving Average) for better generalization
        # Will be initialized after first forward pass
        self.ema = None
        self.use_ema = getattr(cfg, "use_ema", True)  # Enable by default
        self.ema_decay = getattr(cfg, "ema_decay", 0.999)

        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            if 'epoch=' not in self.cfg.pretrained_model_path:
                ckpt.pop('decoder.embed.0.weight')
                ckpt.pop('decoder.output_layer.weight')
                ckpt.pop('decoder.output_layer.bias')
                ckpt.pop('ctc.ctc_lo.weight')
                ckpt.pop('ctc.ctc_lo.bias')

            if self.cfg.transfer_frontend:
                # Support both formats: ckpt with "model_state_dict" key, or flat state_dict
                src = ckpt.get("model_state_dict", ckpt)
                tmp_ckpt = {}
                for k, v in src.items():
                    if k.startswith("trunk.") or k.startswith("frontend3D."):
                        tmp_ckpt[k] = v
                    elif k.startswith("encoder.frontend."):
                        new_k = k.replace("encoder.frontend.", "")
                        tmp_ckpt[new_k] = v
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt, strict=False)

    def configure_optimizers(self):
        # Separate parameter groups so we can use different learning rates for encoder/decoder
        encoder_params = [p for n, p in self.model.named_parameters() if "encoder" in n and p.requires_grad]
        decoder_params = [p for n, p in self.model.named_parameters() if "decoder" in n and p.requires_grad]
        other_params = [p for n, p in self.model.named_parameters() if ("encoder" not in n and "decoder" not in n) and p.requires_grad]

        encoder_lr = getattr(self.cfg.optimizer, "encoder_lr", self.cfg.optimizer.lr)  # Same as decoder
        decoder_lr = getattr(self.cfg.optimizer, "decoder_lr", self.cfg.optimizer.lr)

        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": decoder_lr, "name": "decoder"})
        if other_params:
            param_groups.append({"params": other_params, "lr": decoder_lr, "name": "other"})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))

        # Use Cosine Annealing with Warm Restarts
        T_0 = getattr(self.cfg.optimizer, "T_0", 10)  # First cycle: 10 epochs
        T_mult = getattr(self.cfg.optimizer, "T_mult", 2)  # Cycle multiplier
        eta_min_ratio = getattr(self.cfg.optimizer, "eta_min_ratio", 0.01)

        scheduler = WarmupCosineRestartScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            T_0,
            len(self.trainer.datamodule.train_dataloader()),
            T_mult=T_mult,
            eta_min_ratio=eta_min_ratio
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, step_type="train")

        # EMA: Only enable after epoch 10 (when model has stabilized)
        ema_start_epoch = 10
        if self.use_ema and self.current_epoch >= ema_start_epoch:
            if self.ema is None:
                self.ema = EMA(self.model, decay=self.ema_decay)
                print(f"  ✓ EMA initialized at epoch {self.current_epoch} with decay={self.ema_decay}")
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def on_validation_epoch_start(self):
        """Apply EMA weights before validation"""
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow()

    def on_validation_epoch_end(self):
        """Restore original weights after validation"""
        if self.use_ema and self.ema is not None:
            self.ema.restore()

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        #print(enc_feat.shape)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))

        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = sample["target"]

        actual = self.text_transform.post_process(token_id)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())

        current_cer = compute_cer(predicted, actual)
        current_wer = compute_wer(predicted, actual)

        self.accumulate_cer += current_cer
        self.accumulate_wer += current_wer
        self.batch_num += 1

        return

    def _step(self, batch, batch_idx, step_type):
        # Updated to unpack 5 return values (added enc_feat for alignment)
        loss, loss_ctc, loss_att, acc, enc_feat = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        # Monitor CTC blank probability (collapse detection)
        try:
            if enc_feat is not None and self.model.ctc is not None:
                ctc_logits = self.model.ctc.ctc_lo(enc_feat[0])  # [T, V]
                probs = torch.softmax(ctc_logits, dim=-1)
                blank_prob = probs[:, 0].mean()
                self.log("ctc_blank_prob", blank_prob, on_step=False, on_epoch=True, prog_bar=True)
        except Exception:
            pass


        # Idea 1: Semantic Alignment Loss (Contrastive Learning)
        if step_type == "train":
            # 1. Get Text Embeddings from Decoder
            # Use ground truth targets to get text embeddings
            # targets: [Batch, Length]
            ys_in_pad, _ = add_sos_eos(batch["targets"], self.model.sos, self.model.eos, self.model.ignore_id)
            # [Batch, Length, Dim]
            text_emb = self.model.decoder.embed(ys_in_pad)
            
            # 2. Pooling (Mean Pooling) to get sentence-level representation
            # Mask out padding for correct mean
            # enc_feat: [Batch, Time, Dim]
            vis_mean = enc_feat.mean(dim=1)
            text_mean = text_emb.mean(dim=1)
            
            # 3. Projection
            v_proj = torch.nn.functional.normalize(self.align_proj_visual(vis_mean), dim=1)
            t_proj = torch.nn.functional.normalize(self.align_proj_text(text_mean), dim=1)
            
            # 4. Contrastive Loss (InfoNCE-like)
            # Cosine similarity matrix [Batch, Batch]
            logits = torch.matmul(v_proj, t_proj.T) / self.align_temperature
            labels = torch.arange(batch_size).to(logits.device)
            
            loss_align_v2t = torch.nn.functional.cross_entropy(logits, labels)
            loss_align_t2v = torch.nn.functional.cross_entropy(logits.T, labels)
            loss_align = (loss_align_v2t + loss_align_t2v) / 2

            # Contrastive Learning: Only enable after epoch 20 (when model has learned basics)
            # Then gradually increase weight from 0.0 to 0.1 over 10 epochs
            start_epoch = 20
            warmup_epochs = 10
            target_weight = 0.1

            if self.current_epoch < start_epoch:
                align_weight = 0.0  # Disabled in early training
            elif self.current_epoch < start_epoch + warmup_epochs:
                progress = (self.current_epoch - start_epoch) / warmup_epochs
                align_weight = target_weight * progress
            else:
                align_weight = target_weight

            loss = loss + align_weight * loss_align
            
            self.log("loss_align", loss_align, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("align_weight", align_weight, on_step=False, on_epoch=True, batch_size=batch_size)

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        dataloader = self.trainer.train_dataloader
        if hasattr(dataloader, "batch_sampler"):
            sampler = dataloader.batch_sampler
        elif hasattr(dataloader, "loaders") and hasattr(dataloader.loaders, "batch_sampler"):
            sampler = dataloader.loaders.batch_sampler
        else:
            sampler = None
            
        if sampler and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform_CCS()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        self.accumulate_cer = 0
        self.accumulate_wer = 0
        self.batch_num = 0

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)
        self.log("CCS_cer", self.accumulate_cer / self.batch_num)
        self.log("CCS_wer", self.accumulate_wer / self.batch_num)


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40, length_penalty=0.0):
    # Ensure beam_size doesn't exceed vocabulary size
    vocab_size = len(token_list)
    if beam_size > vocab_size:
        beam_size = vocab_size

    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": length_penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
