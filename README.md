# Evolution Strategies for Atari games

Ce projet a été réalisé dans le cadre de l'unité d'enseignement Intelligence Artificielle pour la Robotique dispensée à Sorbonne Université, Paris VI.

L'objet était premièrement de reproduire les résultats obtenus par Patryk Chrabaszcz, Ilya Loshchilov et Frank Hutter dans l'article _Back to Basics: Benchmarking Canonical Evolution Strategies for Playing Atari_ avec un budget plus limité et des réseaux de neurones de plus petite taille que ceux utilisés par les auteurs. Il s'agissait dans un second temps d'étendre la comparaison OpenAI ES vs CanonicalES à d’autres algorithmes ES tels que CEM ou CMA-ES. Nous avons choisi de tester et comparer ces différents algorithmes sur le jeu Pong et avons obtenu le résultat suivant avec OpenAI ES :



### Prérequis

Si vous souhaitez uniquement visualiser le comportement des intelligences artificielles entraînées sur les différents jeux Atari, vous aurez besoin de :

- Python 3.5
- tensorflow (version 1.14.0) 
- natsort
- gym[atari]

En revanche, l'entraînement des intelligences artificielles nécessitant d'importantes ressources computationnelles, il vous faudra utiliser **Google Cloud Platform** (ou une plateforme similaire) et y créer une machine virtuelle si vous souhaitez réaliser vos propres entraînements. Il vous faudra également installer le module mpi4py.

### Installations

Tensorflow (version 1.14.0) :
```
sudo python3.5 -m pip install tensorflow==1.14.0
```
natsort :
```
sudo python3.5 -m pip install natsort
```
mpi4py :
```
sudo apt-get install python3-mpi4py
```
gym[atari] :
```
sudo python3.5 -m pip install gym[atari]
```

### Lancer les tests
Afin de lancer les différents algorithmes, vous pouvez procéder de la manière suivante:

#### Canonical ES et OpenAI ES
Placer-vous dans le dossier openai_es_canonical_es et choisissez l'algorithme que voulez tester en modifiant votre fichier de configuration (par exemple ./configurations/sample_configuration.json) en changeant l'attribut "optimizer" à "CanonicalESOptimizer" pour le Canonical ES et "OpenAIOptimizer" pour le OpenAI ES.
```
cd openai_es_canonical_es
```
Lancez ensuite l'exécution du fichier main.py de la manière suivante:
```
python3 main.py -e number_of_episodes_per_cpu -g game -c path_to_configuration_file -r name_of_the_run
```

#### CMA-ES
Placez-vous dans le dossier cma_es 
```
cd cma_es
```
Lancez ensuite l'exécution du fichier RL.py
```
python RL.py
```


#### CEM et DQN
Pour lancer l'un des deux algorithmes, placez-vous dans le dossier dqn_cem
```
cd dqn_cem
```
Lancez ensuite l'exécution du fichier run.py en spécifiant le mode (train ou test), le jeu, l'agent (dqn ou cem)...
Ci-dessous un exemple de commande:
```
python ./run.py \
  --mode train \
  --model dqn_atari \
  --agent dqn \
  --game Pong-v0 \
  --base_dir exp \
  --save_weight_interval 250000 \
  --save_log_interval 100 \
  --max_steps 10000000 \
  --memory_limit 1000000 \
  --memory_window_length 4 \
  --visualize_train true \
```

