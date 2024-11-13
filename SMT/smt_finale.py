import subprocess
import json

for i in range(1,21):
    flag = False
    try:
        subprocess.call(["python3", "smt_subprocess.py", str(i)], timeout=300)
    except subprocess.TimeoutExpired:
        print("TIMEOUT: ",i)
        flag = True

    if flag:
        result_dict = {
            "time" : 300,
            "optimal" : False,
            "obj" : 0,
            "sol" : []
        }
        with open(f"res/res_{i}.json","w") as f:
            json.dump(result_dict, f)
    else:
        with open("tmp_output.json", "r") as f:
            file = f.readlines()[0]

        print(f"Instance {i}: {file}")

        with open(f"res/res_{i}.json","w") as f:
            f.write(file)