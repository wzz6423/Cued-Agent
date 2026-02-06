import os
import torch
from omegaconf import OmegaConf
import hydra

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    from lightning_CCS import ModelModule_CCS, get_beam_search_decoder
    from datamodule.av_dataset_CCS import AVDataset_CCS
    from datamodule.transforms import VideoTransform

    print("Loading model...")
    model = ModelModule_CCS(cfg)

    ckpt_path = "/home/ubuntu/wzz/Cued-Agent/lip_agent_and_prompt_decoding_agent/results/test_run_medium/epoch=142-loss_val=52.4193229675293.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cuda:0")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    # Initialize beam search
    beam_search = get_beam_search_decoder(model.model, model.token_list, ctc_weight=0.0, beam_size=10)

    # Load a few test samples
    ds_args = cfg.data.dataset
    test_ds = AVDataset_CCS(
        root_dir=ds_args.root_dir,
        label_path=os.path.join(ds_args.root_dir, ds_args.label_dir, ds_args.test_file),
        subset="test",
        modality=cfg.data.modality,
        audio_transform=None,
        video_transform=VideoTransform("test"),
    )

    print(f"\nTesting on {len(test_ds)} samples")
    print("=" * 60)

    # Test first 5 samples
    for i in range(min(5, len(test_ds))):
        sample = test_ds[i]

        with torch.no_grad():
            enc_feat, _ = model.model.encoder(
                sample["input"].unsqueeze(0).cuda(), None
            )
            enc_feat = enc_feat.squeeze(0)

            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
            print(f"  Raw hyps: {nbest_hyps[0]['yseq']}")
            predicted_ids = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))

            predicted = model.text_transform.post_process(predicted_ids).replace("<eos>", "")
            actual = model.text_transform.post_process(sample["target"])

        print(f"\nSample {i+1}:")
        print(f"  Actual:    '{actual}'")
        print(f"  Predicted: '{predicted}'")

if __name__ == "__main__":
    main()
