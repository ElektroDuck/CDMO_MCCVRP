import subprocess
import json
#python3 main.py --method SMT --model gurobi --solver gecode --instance 1-21  --timeout 300 --update_json True

with open("all_inst.json", "r") as f:
    MODELS = json.load(f)

if __name__ == "__main__":
    for encoding, models in MODELS.items():

        if encoding == "instances":
            continue
        print(f"RUNNING THE {encoding} ENCODINGS")

        for model_name, model_parameters in models["Models"].items():
            if len(model_parameters) > 0:
                for solver in model_parameters:
                    print(f"RUNNING {model_name} with {solver}")
                    subprocess.call([
                        "python3", "main.py",
                        "--method", encoding,
                        "--model", model_name,
                        "--solver", solver,
                        "--instance", MODELS["instances"],
                        "--timeout", "300",
                        "--update_json", "True"
                    ])
            else:
                print(f"RUNNING {model_name}")
                subprocess.call([
                    "python3", "main.py",
                    "--method", encoding,
                    "--model", model_name,
                    "--instance", MODELS["instances"],
                    "--timeout", "300",
                    "--update_json", "True"
                ])


