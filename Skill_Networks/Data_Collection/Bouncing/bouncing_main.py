import yaml
import subprocess

config_filename = "bouncing_config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    if config["WITHOUT PERCEPTION"] or config["PERCEPTION"]:
        subprocess.run(["python", "bouncing.py"], check=True)
        print("DONE!!")
    else:
        print("DONE!!")

