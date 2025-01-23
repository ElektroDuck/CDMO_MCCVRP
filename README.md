# 🚚 CDMO_MCCVRP

Welcome to the **Multi Courier Capacitated Vehicle Routing Problem** (MCCVRP) repository for the **CDMO-2024 exam**! 🎓 This repository contains the code and the project report.

---

## 📋 Quick Reference: Running the Final Models
This section serves as a quick reference guide with ready-to-use commands for running models. All commands have been tested and are guaranteed to work within the Docker environment. 

### CP
Optimized for small istances (1-10)
```bash
python3 main.py --method CP --model Model_A_opt_1_10.mzn --solver gecode --instance 1-21 --timeout 300 --update_json n
```
Optimized for big istances (11-21)
```bash
python3 main.py --method CP --model Model_A_opt_11_21.mzn --solver gecode --instance 1-21 --timeout 300 --update_json n
```
Chuffed
```bash
python3 main.py --method CP --model Model_A_opt_chuffed.mzn --solver chuffed --instance 1-21 --timeout 300 --update_json n
```
No simmetry breaking constraints 
```bash
python3 main.py --method CP --model Model_A_no_sb.mzn --instance 11-21 --solver gecode --timeout 300 --update_json y --use_sb y
```
Base model (without global constraints) 
```bash
python3 main.py --method CP --model Model_A.mzn --solver gecode --instance 1-21 --timeout 300 --update_json n
```

### SMT
base model
```bash
python3 main.py --method SMT --model base --instance 1-21 --timeout 300 --update_json n
```

aux variable model
```bash
python3 main.py --method SMT --model aux_variable --instance 1-21 --timeout 300 --update_json n
```

### SAT
Sequential search
```bash
python3 main.py --method SAT --model Base --instance 1-21 --timeout 300 --update_json n
```
Binary search
```bash
python3 main.py --method SAT --model Binary --instance 1-21 --timeout 300 --update_json n
```

### MIP
Gurobi model
```bash
python3 main.py --method MIP --model gurobi --instance 1-21 --timeout 300 --update_json n
```
Minizinc model
```bash
python3 main.py --method MIP --model minizinc --instance 1-21 --timeout 300 --update_json n
```
## Running the solution checker

Inside the root directory, simply run the command
```bash
 python3 check_solution.py Instances/ res/
```

## 🚀 How to Run the Code

**IMPORTANT**: in order to run the MIP gurobi solver it is necessar to add a gurobi licence. This can be done by modifying the function `get_gurobi_env()` in the file `solvers.py`.

### 🐳 Docker (Recommended)
Using Docker is the preferred method to run this project, as it has been extensively tested. Follow these steps to get started:

#### Step 1: Build the Docker Container
Run the following command in the project root to build the Docker container.  
If you don’t have Docker installed, refer to the official [Docker installation guide](https://docs.docker.com/engine/install/).

```bash
docker build -t cdmo_project .
```

#### Step 2: Start the Container Terminal
Run the command below to start the container and access its terminal:

```bash
docker run -it cdmo_project
```

You will now be inside the container terminal and can execute commands as if working on a virtual machine.

**Example Command:**  
Run the `CP` model `Model_A_opt_1_10.mzn` on all instances using the `gecode` solver with a timeout of 300 seconds:

```bash
python3 main.py --method CP --model Model_A_opt_1_10.mzn --solver gecode --instance 1-21 --timeout 300 --update_json n
```

### Where are my results?

All the output of the models are stored inside the directory `res`. Each model will write its result in a subdirectory contained in `res` (for example, the CP results are located in `res/CP`). Inside a result folder, there are many `json` file that can be consulted in the terminal using the command `cat`.

For example, let's say I want to check the results for the first instance of the MIP models, I can write in the terminal the command

```bash
cat res/MIP/1.json
```

---

### 🖥️ Virtual Environment
You can also run the code on your local machine by setting up a virtual environment. Follow these steps based on your operating system. This option is not reccomendend since it has only been used during development:

#### On Linux
1. Create the virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Verify the installation:
   ```bash
   which python
   ```
   The output should be similar to `.venv/bin/python`.

#### On Windows
1. Create the virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   .venv\Scripts\activate
   ```

3. Verify the installation:
   ```bash
   where python
   ```
   The output should be similar to `.venv\Scripts\python`.

Then, install the dependecies by running
```bash
pip install -r requirements.txt
```
Now you can run the code!

---

### ⚙️ (Additional step) Add MiniZinc to PATH on Windows
Follow the [MiniZinc installation guide](https://docs.minizinc.dev/en/stable/installation.html#microsoft-windows) to add MiniZinc to your system PATH.  

To set the PATH variable, use the following command in the Command Prompt (CMD):

```cmd
setx PATH "%PATH%;YOUR_INSTALLATION_FOLDER"
```

Replace `YOUR_INSTALLATION_FOLDER` with the MiniZinc installation path, such as:

```cmd
C:\Program Files\MiniZinc
```


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
python main.py --method selected_method --model selected_model --instance instance_number --solver selected_solver --timeout timeout_duration --update_json y
```

### Flags

| Flag            | Default | Type  | Required | Description                                                                                          |
|------------------|---------|-------|---------|------------------------------------------------------------------------------------------------------|
| `--method`       | -       | str   | Yes     | The method to use (`CP`, `SMT`, `MIP`)                                                               |
| `--model`        | -       | str   | Yes     | The name of the model to use, based on the chosen method.                                            |
| `--instance`     | 1       | str   | Yes     | Instance to solve: single (`--instance 1`), range (`--instance 1-10`), or all (`--instance all`).    |
| `--solver`       | -       | str   | No      | Solver to use, based on the method.                                                                  |
| `--timeout`      | 300     | int   | No      | Timeout in seconds.                                                                                  |
| `--update_json`  | y       | str   | No      | If y ypdate the solutions in the JSON file. Otherwise type n.                                        |
| `--use_sb`       | y       | str   | No      | If y add the sb constraint in the cp model when the matrix is symmetric. Otherwise type n.           |

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

