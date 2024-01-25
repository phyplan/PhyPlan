import yaml
import subprocess

config_filename = "throwing_config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    if config["PERCEPTION"]:
        subprocess.run(["python", "throwing.py"], check=True)
        subprocess.run(["python", "throwing_perception.py"], check=True)
        print("DONE!!")
    else:
        if config["WITHOUT PERCEPTION"]:
            subprocess.run(["python", "throwing.py"], check=True)
            print("DONE!!")
        else:
            print("DONE!!")

