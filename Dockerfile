FROM minizinc/minizinc:latest

WORKDIR ./CDMO_MCCVRP

COPY . .

RUN apt-get update 
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt install python3.12-venv -y
#CMD python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -r requirements.txt

RUN python3 -m venv .venv
RUN . .venv/bin/activate && python3 -m pip install -r requirements.txt

CMD ["/bin/bash", "-c", "source .venv/bin/activate && exec /bin/bash"]
