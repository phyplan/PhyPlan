import yaml
import subprocess
import numpy as np

config_filename = "sliding_config.yaml"

with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    if config["PERCEPTION"]:
        while (True):
            subprocess.run(["python", "sliding.py"], check=False)
            subprocess.run(["python", "sliding_perception.py"], check=False)
            a = np.loadtxt(config["PERCEPTION_SAVE_DIR"], delimiter=',')
            if a.shape[0] >= config["DATA_LIM"]:
                break
            subprocess.run(['rm', '-rf', config["IMAGES"]], check=True)
        print("DONE!!")
    else:
        if config["WITHOUT PERCEPTION"]:
            while(True):
                subprocess.run(["python", "sliding.py"], check=False)
                a = np.loadtxt(config["WITHOUT_PERCEPTION_SAVE_DIR"], delimiter=',')
                if a.shape[0] >= config["DATA_LIM"]:
                    break
            print("DONE!!")
        else:
            print("DONE!!")
