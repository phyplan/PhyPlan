import yaml
import subprocess

config_filename = "swinging_config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    if config["PERCEPTION"]:
        subprocess.run(["python", "swinging.py"], check=True)
        subprocess.run(["python", "swinging_perception.py"], check=True)
        print("DONE!!")
    else:
        if config["WITHOUT PERCEPTION"]:
            subprocess.run(["python", "swinging.py"], check=True)
            print("DONE!!")
        else:
            print("DONE!!")

