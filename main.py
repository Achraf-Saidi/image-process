import argparse, os, yaml, torch, random, numpy as np
from termcolor import cprint
from Networks.model import SupervisedExperiment, SSLExperiment

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", required=True, help="YAML name (without .yaml) in Todo_List/")
    args = parser.parse_args()

    cfg_path = os.path.join("Todo_List", f"{args.exp}.yaml")
    assert os.path.isfile(cfg_path), f"Config not found: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("Results", exist_ok=True)
    exp_name = cfg.get("EXP_NAME", args.exp)
    out_dir  = os.path.join("Results", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.get("SEED", 42))
    device = torch.device(cfg.get("DEVICE", "cpu"))

    mode = cfg.get("MODE", "supervised")
    cprint(f"[EXP] {exp_name} | MODE={mode} | DEVICE={device}", "cyan")

    if mode == "supervised":
        SupervisedExperiment(cfg, out_dir, device).run()
    elif mode == "ssl":
        SSLExperiment(cfg, out_dir, device).run()
    else:
        raise ValueError("MODE must be 'supervised' or 'ssl'")

if __name__ == "__main__":
    main()
