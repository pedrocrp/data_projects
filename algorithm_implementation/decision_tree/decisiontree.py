import numpy as np
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, predicted_class):
        """
        Cria um nó da árvore de decisão.

        Args:
            predicted_class: A classe que este nó prevê.
        """
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    
    def is_leaf(self):
        """
        Verifica se este nó é uma folha (ou seja, não tem filhos).
        """
        return self.left is None and self.right is None

    
    def has_left(self):
        """
        Verifica se este nó tem um filho à esquerda.
        """
        return self.left is not None

    
    def has_right(self):
        """
        Verifica se este nó tem um filho à direita.
        """
        return self.right is not None

    
    def has_both_children(self):
        """
        Verifica se este nó tem ambos os filhos (esquerdo e direito).
        """
        return self.has_left() and self.has_right()

    
    def get_leaves(self):
        """
        Obtém todas as folhas da sub-árvore enraizada neste nó.
        """
        if self.is_leaf():
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    
    def make_leaf(self):
        """
        Transforma este nó em uma folha, removendo todos os seus filhos.
        """
        self.left = None
        self.right = None

    
    def restore(self, leaves):
        """
        Restaura os filhos deste nó a partir de uma lista de folhas.
        """
        self.left = leaves[0]
        self.right = leaves[1]


class DecisionTree:
    def __init__(self, max_depth=None):
        """
        Cria uma árvore de decisão.

        Args:
            max_depth: A profundidade máxima da árvore.
        """
        self.max_depth = max_depth


    def fit(self, X, y):
        """
        Treina a árvore de decisão.

        Args:
            X: A matriz de características.
            y: O vetor de classes.
        """
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        """
        Cresce a árvore de decisão de forma recursiva.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node


    def _best_split(self, X, y):
        """
        Esta função identifica a melhor divisão para os dados no nó, utilizando o critério de ganho de informação. 

        Primeiro, verifica se o tamanho dos dados é menor ou igual a 1, caso positivo, não há necessidade de dividir, então retorna None, None.

        Caso contrário, inicia o processo de divisão. Para cada característica nos dados, calcula o limite e a classe correspondente em ordem crescente de valores. 

        Com esses valores, ele cria duas listas, num_left e num_right, que contêm a contagem de cada classe nos dados que estão à esquerda e à direita do limite, respectivamente. 

        Em seguida, calcula o ganho de informação para cada possível ponto de divisão. O ponto de divisão que resulta no maior ganho de informação é selecionado como o melhor ponto de divisão. 

        A função então retorna o índice da característica e o valor do limiar para esse melhor ponto de divisão.
        """
        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gain = 0.0
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            split_entropy = self._calculate_split_entropy(X[:, idx])

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                gain = 1.0 - gini
                gain_ratio = gain / split_entropy

                if gain_ratio > best_gain:
                    best_gain = gain_ratio
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr


    def predict(self, X):
        """
        Faz previsões para um conjunto de dados de entrada.
        """
        return [self._predict(inputs) for inputs in X]


    def _predict(self, inputs):
        """
        Faz uma previsão para uma única instância de entrada.
        """
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


    def prune(self, X_test, y_test):
        """
        Realiza a poda da árvore de decisão.
        """
        self.tree_ = self._prune(self.tree_, X_test, y_test)

    
    def _prune(self, node, X_test, y_test):
        """
        Realiza a poda da árvore de decisão de forma recursiva.
        """
        if node.is_leaf():
            return node
        if node.has_left():
            node.left = self._prune(node.left, X_test, y_test)
        if node.has_right():
            node.right = self._prune(node.right, X_test, y_test)
        if node.has_both_children():
            leaves = node.get_leaves()
            original_accuracy = self.calculate_accuracy(y_test, self.predict(X_test))
            node.make_leaf()
            new_accuracy = self.calculate_accuracy(y_test, self.predict(X_test))
            if new_accuracy >= original_accuracy:
                return node
            node.restore(leaves)
        return node

    
    def calculate_accuracy(self, y_true, y_pred):
        """
        Calcula a acurácia de um conjunto de previsões.

        Parâmetros:
            y_true: As classes verdadeiras.
            y_pred: As classes previstas.
        """
        return accuracy_score(y_true, y_pred)


    def _calculate_split_entropy(self, X):
        """
        Esta função calcula a entropia de uma divisão.

        A entropia é uma métrica que mede a impureza de uma divisão. Quanto menor a entropia, mais "puro" é o conjunto de dados.
        """
        _, counts = np.unique(X, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(-p * np.log2(p) for p in probabilities)
        return entropy







from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Carregar o conjunto de dados iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar a árvore de decisão
tree = DecisionTree(max_depth=3)

# Treinar a árvore de decisão
tree.fit(X_train, y_train)

# Realizar previsões no conjunto de teste
predictions = tree.predict(X_test)

# Calcular a acurácia
accuracy = tree.calculate_accuracy(y_test, predictions)
print(f'Acurácia: {accuracy*100:.2f}%')

# Poda da árvore de decisão
tree.prune(X_test, y_test)

# Realizar previsões após a poda
predictions_pruned = tree.predict(X_test)

# Calcular a acurácia após a poda
accuracy_pruned = tree.calculate_accuracy(y_test, predictions_pruned)
print(f'Acurácia após a poda: {accuracy_pruned*100:.2f}%')



clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

