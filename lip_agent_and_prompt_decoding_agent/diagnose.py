import os
import torch
import hydra

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    from lightning_CCS import ModelModule_CCS, get_beam_search_decoder
    from datamodule.av_dataset_CCS import AVDataset_CCS
    from datamodule.transforms import VideoTransform

    print("=== Diagnostic Test ===\n")
    model = ModelModule_CCS(cfg)

    ckpt_path = "results/test_run_medium/epoch=142-loss_val=52.4193229675293.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cuda:0")

    # Check state dict keys match
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint['state_dict'].keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    if not missing and not unexpected:
        print("All keys match.")

    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    # Load one sample
    ds_args = cfg.data.dataset
    test_ds = AVDataset_CCS(
        root_dir=ds_args.root_dir,
        label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.test_file),
        subset="test", modality=cfg.data.modality,
        audio_transform=None, video_transform=VideoTransform("test"),
    )

    sample = test_ds[0]
    inp = sample["input"].unsqueeze(0).cuda()

    with torch.no_grad():
        # 1) Check encoder output
        enc_feat, _ = model.model.encoder(inp, None)
        print(f"\nEncoder output shape: {enc_feat.shape}")
        print(f"Encoder output stats: mean={enc_feat.mean():.4f}, std={enc_feat.std():.4f}")

        # 2) Check CTC output
        ctc_out = model.model.ctc.ctc_lo(enc_feat)
        ctc_probs = torch.softmax(ctc_out, dim=-1)
        blank_prob = ctc_probs[:, :, 0].mean().item()
        print(f"\nCTC blank prob (avg): {blank_prob:.4f}")
        print(f"CTC top tokens per frame (first 5 frames):")
        for t in range(min(5, ctc_probs.shape[1])):
            topk = ctc_probs[0, t].topk(3)
            print(f"  Frame {t}: {list(zip(topk.indices.tolist(), [f'{v:.3f}' for v in topk.values.tolist()]))}")

        # 3) Check decoder with greedy (teacher forcing with sos)
        enc_feat_sq = enc_feat.squeeze(0)
        sos = model.model.sos
        eos = model.model.eos
        print(f"\nsos={sos}, eos={eos}")

        # 4) Try beam search with different ctc weights
        for cw in [0.0, 0.1, 0.5]:
            beam_search = get_beam_search_decoder(model.model, model.token_list, ctc_weight=cw, beam_size=10)
            nbest = beam_search(enc_feat_sq)
            hyp = [h.asdict() for h in nbest[:1]]
            seq = hyp[0]["yseq"]
            score = hyp[0]["score"]
            print(f"\nctc_weight={cw}: yseq={seq}, score={score:.4f}")

if __name__ == "__main__":
    main()
