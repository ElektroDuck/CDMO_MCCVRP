import subprocess
import json

for i in range(1,21):
    # Running throught all the instances
    flag = False
    try:
        # Calling the SMT solver as a subprocess, so we can apply timeout
        subprocess.call(["python3", "smt_subprocess.py", str(i)], timeout=300)
    except subprocess.TimeoutExpired:
        print("TIMEOUT: ",i)
        flag = True

    # Saving results
    if flag:
        try:
            with open("tmp_output.json", "r") as f:
                file = f.readlines()[0]
            result_dict = json.loads(file)
            result_dict["time"] = 300
            result_dict["optimal"] = False
        except:
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