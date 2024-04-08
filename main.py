import random

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from Evolution.EvolutionaryLib import BaseChromosome, ChromosomeClassFactory
import pandas as pd


class DefaultChromosome(BaseChromosome):
    def __init__(self):
        super().__init__()

    def crossover(self, other):
        new = DefaultChromosome()
        self.copy(new)

        for gene in self:
            gene = gene.strip('_')
            new[gene] = random.choice([self[gene], other[gene]])

        return new

    def mutate(self, rate=0.3):
        new = DefaultChromosome()
        self.copy(new)

        for gene in self:
            gene = gene.strip('_')
            new[gene] = new['_' + gene].get() if random.random() < rate else self[gene]

        return new


def evaluate(population: list[DefaultChromosome], dataset: tuple, consntructor) -> None:
    for ch in population:
        if ch.get_score() is None:
            model = consntructor(ch)
            model.fit(dataset[0], dataset[2])
            ch.set_score(accuracy_score(dataset[3], model.predict(dataset[1])))


def decision_tree_constructor(chromosome: DefaultChromosome):
    return DecisionTreeClassifier(criterion=chromosome.criterion,
                                  splitter=chromosome.splitter,
                                  max_depth=chromosome.max_depth)


def random_forest_constructor(chromosome: DefaultChromosome):
    return RandomForestClassifier(n_estimators=chromosome.n_trees,
                                  max_depth=chromosome.max_depth)


class Web():
    def __init__(self):
        self.models = {
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'parameters': {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': range(2, 50)
                },
                'constructor': decision_tree_constructor
            },

            'random_forest': {
                'model': RandomForestClassifier,
                'parameters': {
                    'max_depth': range(2, 50),
                    'n_trees': range(3, 300)
                },
                'constructor': random_forest_constructor
            },
        }

        self.datasets = {
            'iris': self.read_iris,
            'breast_cancer': 2,
        }

        self.model = None
        self.model_params = None
        self.dataset = None
        self.evolution = None

        st.set_page_config(page_title="Sklearn classifiers")
        self.choose_dataset()
        self.input_model()
        self.input_evolution_params()

        self.start_evolution()

    def input_model(self):
        self.model = st.radio("Choose model", self.models.keys())

        if not self.model:
            return

        st.markdown("""---""")
        st.markdown("Model parameters selection")
        model_params = self.models[self.model]['parameters']

        chosen_params = dict()

        for parameter in model_params:
            parameter_name = f'{parameter}'

            if type(model_params[parameter]) is range:
                chosen_params[parameter] = {
                    'data': st.slider(parameter_name, min_value=model_params[parameter].start,
                                      max_value=model_params[parameter].stop,
                                      value=(
                                          model_params[parameter].start,
                                          model_params[parameter].stop),
                                      step=model_params[parameter].step),
                    'step': model_params[parameter].step,
                    'type': int
                }



            elif type(model_params[parameter]) is list:
                chosen_params[parameter] = st.multiselect(parameter_name, tuple(model_params[parameter]))

            elif type(model_params[parameter]) is tuple:
                chosen_params[parameter] = {
                    'data': st.slider(parameter_name, min_value=float(model_params[parameter][0]),
                                      max_value=float(model_params[parameter][1]),
                                      value=(float(model_params[parameter][0]),
                                             float(model_params[parameter][1])),
                                      step=0.01),
                    'type': float
                }
        self.model_params = chosen_params

    def input_evolution_params(self):
        st.markdown("""---""")
        st.markdown("Evolution parameters selection")
        population_size = st.slider("Population size", min_value=6,
                                    max_value=300,
                                    value=30,
                                    step=3)

        epochs = st.slider("Epochs number", min_value=3,
                           max_value=1000,
                           step=1)

        self.evolution = {
            'population_size': population_size,
            'epochs': epochs
        }

    def start_evolution(self):
        if not st.button("Start evolution"):
            return

        if not (self.model and self.model_params and self.dataset and self.evolution):
            return

        parameters = dict()
        for p, v in self.model_params.items():
            if type(v) is list:
                parameters[p] = v
            elif type(v) is dict:
                if v['type'] is int:
                    parameters[p] = list(range(v['data'][0], v['data'][1] + 1, v['step']))
                elif v['type'] is float:
                    parameters[p] = v['data']
        factory = ChromosomeClassFactory(**parameters)

        population = [factory.generate(DefaultChromosome) for _ in range(self.evolution['population_size'])]
        best = factory.generate(DefaultChromosome)
        best.set_score(-1e10)

        stat_best = []

        progress_bar = st.progress(0, text="Progress:")
        for iteration_number in range(self.evolution['epochs']):
            progress_bar.progress((iteration_number + 1) / self.evolution['epochs'],
                                  text=f"Progress: {(iteration_number + 1)} / {self.evolution['epochs']}\nbest: {best.get_score()}")

            evaluate(population=population, dataset=self.dataset, consntructor=self.models[self.model]['constructor'])
            population = sorted(population, key=DefaultChromosome.get_score,
                                reverse=True)[:self.evolution['population_size'] // 3]

            if best is None or best.get_score() < population[0].get_score():
                best = population[0]

            stat_best.append(best.get_score())

            while len(population) != self.evolution['population_size']:
                r = random.random()
                if r < 0.33:
                    population.append(random.choice(population).crossover(random.choice(population)))
                elif 0.33 <= r < 0.66:
                    population.append(random.choice(population).mutate())
                else:
                    population.append(factory.generate(DefaultChromosome))

        progress_bar.progress(1.0, 'Done!')

        fig, ax = plt.subplots()
        ax.plot(stat_best)
        st.pyplot(fig)

    def choose_dataset(self):
        dataset_key = st.selectbox("Choose dataset", self.datasets.keys())
        self.dataset = self.datasets[dataset_key]()

    def read_iris(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        data = pd.read_csv(url, names=['s_length', 's_width', 'p_length', 'p_width', 'y'])
        data['y'] = data['y'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=42,
                                                            shuffle=True)
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    Web()
