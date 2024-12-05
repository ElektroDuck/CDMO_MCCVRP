# CDMO_MCCVRP
Multi Currier Capacitated Vehicle Routing Problem

## Virtual Enviroment
TODO: Tutta sta parte qua in teoria non serve a niente no? tanto si usa il docker

### LINUX

```python3 -m venv .venv```

```source .venv/bin/activate```

check if the installation is correct

```which python```

the response should be like: .venv/bin/python
WINDOWS

```python3 -m venv .venv```

```.venv\Scripts\activate```

check if the installation is correct

```where python```

the response should be like: .venv\Scripts\python

## Come aggiungere la variabile di minizinc al path windows

la guida sta qui: https://docs.minizinc.dev/en/stable/installation.html#microsoft-windows

é un po un casino ma l'idea é di usare questo comando dal cmd
```C:\>setx PATH "%PATH%;YOUR_INSTALLATION_FOLDER"```

dove la guida dice che YOUR_INSTALLATION_FOLDER dovrebbe essere tipo ```C:\Program Files\MiniZinc 2.8.7 (bundled)```

con me ha funzionato con ```setx PATH "%PATH%;C:\Program Files\MiniZinc```



# Docker Instructions

To run the project correctly, follow these steps:

### Step 1: Build the Docker Container
To build the docker container, execute this command in the root of the project. If Docker is not installed in your system, please follow the official [documentation](https://docs.docker.com/engine/install/).

```bash
docker build -t cdmo_project .
```

### Step 2: Start the Container Terminal

```bash
docker run -it cdmo_project
```

You will now be inside the container's terminal, where you can execute commands as if within a virtual machine.

**Example Command:**
This is an example command to run the `CP` model named `Model_A_gc_corrected_changedoutput.mzn` on the instances `1,3,4` using `gecode` with a timeout of `20` seconds
```bash
python run_model.py --method CP --model Model_A_gc_corrected_changedoutput.mzn --instance 1,3,4 --solver gecode --timeout 20
```

---

# Running All Instances

To run all models with different configurations, use the following script:

```bash
python3 run_all_instances.py
```

This script executes all models described in the report and generates results in `.json` format, saved in the `res/` folder. 

Configurations are loaded from the `all_inst.json` file, which can be modified to define custom executions.

---

# Running Python Code
To run a single model using particular instructions, refer to this section of the readme

### Example Base Command

```bash
python main.py --method selected_method --model selected_model --instance instance_number --solver selected_solver --timeout timeout_duration --update_json True
```

### Flags

| Flag            | Default | Type  | Required | Description                                                                                          |
|------------------|---------|-------|---------|------------------------------------------------------------------------------------------------------|
| `--method`       | -       | str   | Yes     | The method to use (`CP`, `SMT`, `MIP`)                                                              |
| `--model`        | -       | str   | Yes     | The name of the model to use, based on the chosen method.                                            |
| `--instance`     | 1       | str   | Yes     | Instance to solve: single (`--instance 1`), range (`--instance 1-10`), or all (`--instance all`).   |
| `--solver`       | -       | str   | No      | Solver to use, based on the method.                                                                 |
| `--timeout`      | 300     | int   | No      | Timeout in seconds.                                                                                 |
| `--int_res`      | False   | bool  | No      | Display intermediate results (not available for all methods/models).                                |
| `--update_json`  | False   | bool  | No      | Update the solutions in the JSON file.                                                              |

---

## Methods

Detailed explanations of the implemented models are available in `report.pdf`.

### CP

Example Command:

```bash
python main.py --method CP --model Model_A_gc_corrected_changedoutput.mzn --instance all --solver gecode --timeout 300 --update_json True
```

Supported flags:
- `--method`: CP
- `--model`: (Specify your CP models)
- `--solver`: (Specify supported solvers for CP)

---

### SMT

Implemented using the Z3 solver. Example Command:

```bash
python main.py --method SMT --model z3_solver --instance all --timeout 300 --update_json True
```

Supported flags:
- `--method`: SMT
- `--model`: `base` / `aux_variable`

Note: `--solver` and `--int_res` are not available for this method.

---

### MIP

The MIP encoding is available in two variants: Gurobi and MiniZinc. Use the `--model` flag to select one. The MiniZinc variant uses the `gecode` solver.

Example Command:

```bash
python main.py --method MIP --model gurobi --instance all --timeout 300 --update_json True
```

Supported flags:
- `--method`: MIP
- `--model`: `gurobi`, `minizinc`

Note: `--solver` and `--int_res` are not available for this method.








