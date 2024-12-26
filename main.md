# Automating the Search for Artificial Life with Foundation Models

<h1 align="center">
  <a href="https://github.com/SakanaAI/CycleQD/">
    <img width="300" alt="Screenshot 2024-10-16 at 20 35 47" src="https://github.com/user-attachments/assets/bd60128a-7a55-413c-a3d5-640295c5b09b"></a><br>
<b>Automating the Search for Artificial Life with Foundation Models</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2412.17799">[Paper]</a> |
  🤗 <a href="https://huggingface.co/SakanaAI">[Hugging Face]</a> |
  📝 <a href="https://sakana.ai/cycleqd/">[Blog]</a>
</p>

## Installation 

by default, it installs jax 0.4.38 jaxlib-0.4.38 and doesn't work.
we need jax 0.4.31 and jaxlib 0.4.31

```shell
conda create --name asal python=3.10.13





```

### Basic library

```shell
pip install -r requirements.txt
```

### Task evaluator

```shell
cd evaluation/fishfarm
pip install -e .
```


#### VLLM module
 ```shell
 pip install vllm
```

#### Evalplus task
```shell
pip install git+https://github.com/evalplus/evalplus@1895d2f6aa8895044a7cf69defc24bd57695e885
```

#### DBBench task
Run the following commands after docker installation
```shell
docker pull mysql
pip install mysql-connector-python==8.0.32 docker==6.1.2
```
**Tips**
* If you encounter errors connecting to MySQL containers, please increase the value of `/proc/sys/fs/aio-max-nr` (e.g., `echo 1048576 | sudo tee /proc/sys/fs/aio-max-nr`)
* We use `requests==2.31.0` in our setup. (see [this issue](https://github.com/docker/docker-py/issues/3256))

  
#### OSInteraction task
Run the following command after docker installation
```shell
python data/os_interaction/images.py build -c data/os_interaction/configs/std.yaml -r .
```

### Celery and redis  
```shell
pip install celery  
pip install redis
```

## Training

### Single worker

```shell
python3 main.py
```

### Multiple workers

```shell
# 1. Start rabbitmq Broker
docker run -d -p 5672:5672 -v utils/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf rabbitmq
# 2. Start redis Broker
docker run -d -p 6379:6379 -v utils/redis.conf:/etc/redis/redis.conf redis redis-server /etc/redis/redis.conf
# 3. Start Worker(s)
python3 main.py -m celery.mode=worker
# 4. Start Main
python3 main.py -m celery.mode=main
```

## Bibtex

To cite our work, you can use the following:

```

@article{kumar2024asal,
  title = {Automating the Search for Artificial Life with Foundation Models},
  author = {Akarsh Kumar and Chris Lu and Louis Kirsch and Yujin Tang and Kenneth O. Stanley and Phillip Isola and David Ha},
  year = {2024},
  url = {https://pub.sakana.ai/asal/}
}

```





