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

# How to run the code 

Base command 

```python main.py --method selected_method --model selected_model --instance number of the instance --solver selected_solver --timeout your_timeout --update_json True```

## Flags
| Flag        | Default | Type | required | Description                                                  |   |   |   |   |   |
|-------------|---------|------|----------|--------------------------------------------------------------|---|---|---|---|---|
| method      | -       | str  | True     | The method to use (CP, SMT, MIP)                             |   |   |   |   |   |
| model       | -       | str  | True     | The name of the model to use                                 |   |   |   |   |   |
| instance    | 1       | str  | True     | The number of the instances to solve                         |   |   |   |   |   |
| solver      | -       | str  | False    | The solver to use                                            |   |   |   |   |   |
| timeout     | 300     | int  | False    | Timeout time, expressed in seconds                           |   |   |   |   |   |
| int_res     | False   | bool | False    | Show intermediate results, not available for all the systems |   |   |   |   |   |
| update_json | False   | bool | False    | Update the solutions contained in the json file              |   |   |   |   |   |
|             |         |      |          |                                                              |   |   |   |   |   |
|             |         |      |          |                                                              |   |   |   |   |   |


