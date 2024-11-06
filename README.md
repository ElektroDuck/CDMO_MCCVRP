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

