import numpy as np
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, predicted_class):
        """
        Cria um nó da árvore de decisão

        Args:
            predicted_class: A classe que este no preve
        """
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    
    def is_leaf(self):
        """
        Verifica se este nó é uma folha
        """
        return self.left is None and self.right is None

    
    def has_left(self):
        """
        Verifica se este nó tem um filho a esquerda
        """
        return self.left is not None

    
    def has_right(self):
        """
        Verifica se este nó tem um filho a direita
        """
        return self.right is not None

    
    def has_both_children(self):
        """
        Verifica se este nó tem os dois filhos
        """
        return self.has_left() and self.has_right()

    
    def get_leaves(self):
        """
        Obtém todas as folhas da sub-árvore deste nó
        """
        if self.is_leaf():
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    
    def make_leaf(self):
        """
        Transforma este nó em uma folha, removendo todos os seus filhos
        """
        self.left = None
        self.right = None

    
    def restore(self, leaves):
        """
        Restaura os filhos deste nó a partir de uma lista de folhas
        """
        self.left = leaves[0]
        self.right = leaves[1]


class DecisionTree:
    def __init__(self, max_depth=None):
        """
        Cria uma árvore de decisão

        Args:
            max_depth: A profundidade máxima da árvore
        """
        self.max_depth = max_depth


    def fit(self, X, y):
        """
        Treina a árvore de decisão.

        Args:
            X: O dataset com features
            y: O array de classes
        """
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        """
        Constrói a árvore de decisão de forma recursiva
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
        Esta função identifica a melhor divisão para os dados no nó, utilizando o critério de ratio do ganho de informação
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
        Faz previsões para um conjunto de dados de entrada
        """
        return [self._predict(inputs) for inputs in X]


    def _predict(self, inputs):
        """
        Faz uma previsão para uma única instância de entrada
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
        Realiza a poda da árvore de decisão
        """
        self.tree_ = self._prune(self.tree_, X_test, y_test)

    
    def _prune(self, node, X_test, y_test):
        """
        Realiza a poda da árvore de decisão de forma recursiva
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
        Calcula a acurácia de um conjunto de previsões

        Parâmetros:
            y_true: Classificação correta
            y_pred: Classificação do modelo
        """
        return accuracy_score(y_true, y_pred)


    def _calculate_split_entropy(self, X):
        """
        Calcula a entropia de uma divisão no nó
        """
        _, counts = np.unique(X, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(-p * np.log2(p) for p in probabilities)
        return entropy
