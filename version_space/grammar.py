import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

import torch
from torch.distributions.categorical import Categorical

class Grammar:
    def __init__(self, variables, productions, semantics):
        # List of variables in the grammar
        self.idx2variable = variables

        # map from variables to index in idx2variable
        self.variable2idx = {v: i for i, v in enumerate(variables)}

        # List of tuples, with each tuple containing only the 
        # variables on the right hand side of the production
        self.productions = productions

        # List of tuples of functions, where the jth element of a tuple at index
        # i represents the semantics of the jth production with the ith variable as
        # the LHS
        self.semantics = semantics

        # Precomputed mask for sampling programs
        self.mask = torch.ones((max([len(s) for s in semantics]), len(productions)))
        for i, s in enumerate(semantics):
            self.mask[len(s):, i] = -float("inf")

    def execute(self, program_vector, x, y):
        # Start execution at the start node of the program
        return self.__execute_helper(program_vector, x, y, 0)
    
    def __execute_helper(self, program_vector, x, y, idx):
        # Execute the program by recursively computing the values of the arguments to the ith production,
        # and applying the semantics of the ith production to obtain the result.
        args = [self.__execute_helper(program_vector, x, y, self.variable2idx[v]) 
                    for v in self.productions[idx]]
        return self.semantics[idx][program_vector[idx]](x, y, *args)
    
    def sample(self, P):
        """
            Takes a logits P of shape (batch_size x arity x number of productions) and sample programs
            from the distribution. 

            This is done by first masking invalid productions in the program to -inf (so that the probability is zero after applying softmax) using the mask matrix, and then constructing a categorical distribution from which to sample productions. These productions together form a program vector which can be passed to the execute function to obtain the result.
        """
        batch_size = P.shape[0]
        arity = P.shape[1]
        n_productions = P.shape[2]

        assert P.numel() == batch_size * self.mask.numel()

        distribution = Categorical(logits=(P * self.mask).movedim(2, 1).reshape(-1, arity))
        sample = distribution.sample()

        return sample.reshape(batch_size, -1)

def make_constant_function(i):
    return lambda x, y: i

class ShapeGridGrammar(Grammar):
    """
        Program vector represents the program as a list of indices, where the ith
        element of the vector indicates the index of the production of the ith variable below.

        Program -> Shape, Colour
        Shape -> Box(Left, Right, Top, Bottom, Thickness, Outside, Inside)
        Left -> 0 | 1 | 2 | 3 | ... | grid_size
        Right -> 0 | 1 | 2 | 3 | ... | grid_size
        Top -> 0 | 1 | 2 | 3 | ... | grid_size
        Bottom -> 0 | 1 | 2 | 3 | ... | grid_size
        O   -> chicken | pig (or all the shapes in the shapes argument except the last)
        I   -> chicken | pig | pebble (or all the shapes in the shapes argument)
        Thickness -> 1 | 2 | 3
        Colour   -> [red , green , blue][A2(A1)]
        A1 -> x | y | x + y
        A2 -> lambda z:0 | lambda z:1 |lambda z:2 | lambda z:z%2 |lambda z:z%2+1 |lambda z:2*(z%2)
    """

    def __init__(self, grid_size, shapes=['CUBE', 'SPHERE', 'EMPTY'], colours=['R', 'G', 'B']):
        variables = ['Program', 'Shape', 'Left', 'Right', 'Top', 'Bottom', 'Thickness', 'Outside',
                        'Inside', 'Colour', 'A1', 'A2']
        productions = [('Shape', 'Colour'), 
                        ('Left', 'Right', 'Top', 'Bottom', 'Thickness', 'Outside', 'Inside'), 
                        tuple(), tuple(), tuple(), tuple(), tuple(), tuple(), tuple(),
                        ('A1', 'A2'),
                        tuple(), tuple()]
        semantics = [
                        (lambda x, y, shape, colour: (shape, colour),),
                        (lambda x, y, left, right, top, bottom, thickness, outside, inside: 
                            "EMPTY" if (left > x or x > right or top > y or y > bottom)
                                    else (outside if any([
                                        0 <= (x - left) < thickness, 
                                        0 <= (y - top) < thickness, 
                                        0 <= (right - x) < thickness, 
                                        0 <= (bottom - y) < thickness]
                                        )
                                        else inside),),
                        tuple(make_constant_function(i) for i in range(grid_size)), 
                        tuple(make_constant_function(i) for i in range(grid_size)), 
                        tuple(make_constant_function(i) for i in range(grid_size)), 
                        tuple(make_constant_function(i) for i in range(grid_size)),
                        tuple(make_constant_function(i) for i in range(1, 4)),
                        tuple(make_constant_function(shape) for shape in shapes[:-1]),
                        tuple(make_constant_function(shape) for shape in shapes),
                        (lambda x, y, a1, a2: colours[a2(a1)],),
                        (lambda x, y: x, lambda x, y: y, lambda x, y: x + y),
                        (lambda x, y: lambda z: 0, lambda x, y: lambda z: 1, lambda x, y: lambda z: 2, 
                            lambda x, y: lambda z: z % 2, lambda x, y: lambda z: z % 2 + 1, lambda x, y: lambda z: 2 * (z % 2))
                        ]

        super().__init__(variables, productions, semantics)
        self.grid_size = grid_size

def draw(grammar, program_vector, name):
    L = grammar.grid_size
    R = 0.9 / 2 / L
    plt.figure()
    currentAxis = plt.gca(aspect='equal')

    for i in range(grammar.grid_size):
        for j in range(grammar.grid_size):
            shape, color = grammar.execute(program_vector, i, j)
            # x,y = coord
            if shape == 'CUBE':
                currentAxis.add_patch(Rectangle((i/L, j/L), 2*R,2*R, facecolor=color))
            if shape == 'SPHERE':
                currentAxis.add_patch(Circle((i/L+R, j/L+R), R, facecolor=color))

    plt.savefig(f'drawings/{name}.png')
    plt.close()

if __name__ == "__main__":
    grammar = ShapeGridGrammar(7)
    programs = grammar.sample(torch.ones((1, 7, 12))).detach().tolist()
    print(programs)
    draw(grammar, programs[0], 'refactored')