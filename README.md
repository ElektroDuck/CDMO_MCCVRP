# CDMO_MCCVRP
Multi Currier Capacitated Vehicle Routing Problem

## Virtual Enviroment

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


# Docker

build the container
```docker build -t cdmo_project .```

work on the terminal of the container
```docker run -it cdmo_project```

from now on the terminal opens and we can execute commands inside the virtual machine

Example: 
```python run_model.py --method CP --model Model_A_gc_corrected_changedoutput.mzn --instance 1,3,4 --solver gecode --timeout 20```

# How to run the python code 

Example: Base command 

```
python main.py --method selected_method --model selected_model --instance number of the instance --solver selected_solver --timeout your_timeout --update_json True
```

## Flags
| Flag   | Default | Type | required | Description |
|--------|---------|------|----------| ----------- |
| method      | -       | str  | True     | The method to use (`CP`, `SMT`, `MIP`) |
| model       | -       | str  | True     | The name of the model to use, the values depends on the used method, see below |
| instance    | 1       | str  | True     | The number of the instances to solve, can be a specific instance (e.g. `--instance 1`), solve in a range (e.g. `--instance 1-10`), solve all (e.g. `--instance all`) |
| solver      | -       | str  | False    | The solver to use. The values depends on the method, see below |
| timeout     | 300     | int  | False    | Timeout time, expressed in seconds |   
| int_res     | False   | bool | False    | Show intermediate results, not available for all the method and models |  
| update_json | False   | bool | False    | Update the solutions contained in the json file |   

## Mothods 

### CP
```
python main.py --method CP --model Model_A_gc_corrected_changedoutput.mzn --instance all --solver gecode --timeout 300 --update_json True
``` 

The CP method can be runned with the following flags: 
- --method:  CP
- --model: TODO INSERT MODELS 
- --solver: TODO, insert solvers
```

### SMT

```
python main.py --method SMT --model z3_solver --instance all --timeout 300 --update_json True
```

The SMT method can be runned with the following flags: 
- --method:  `SMT`
- --model: `z3_solver`

The falgs `--solver` and  `--int_res` aren't available for this methodology.

### MIP 

The MIP encoding has been proposed with 2 different languages, Gurobi and Minizinc. It is possible to choose which one to use usign the `--model` flag. The Minizinc version is gonna use the `gecode` solver. 

```python main.py --method MIP --model gurobi --instance all --timeout 300 --update_json True```

The MIP method can be runned with the following flags: 
- --method:  `MIP`
- --model: `gurobi`, `minizinc`    

The falgs `--solver` and  `--int_res` aren't available for this methodology.








