import torch
import random

from grammar import ShapeGridGrammar

def sample_S0_shape_grid_spec(grammar, program_vector, min_spec_len=1, max_spec_len=8):
    spec_len = random.randint(min_spec_len, max_spec_len)
    spec_idx = random.sample(range(grammar.grid_size * grammar.grid_size), spec_len)
    spec_x = [i // grammar.grid_size for i in spec_idx]
    spec_y = [i % grammar.grid_size for i in spec_idx]
    spec = list()
    for x, y in zip(spec_x, spec_y):
        s, c = grammar.execute(program_vector, x, y)
        if s == "EMPTY":
            spec.append((x, y, s, "EMPTY"))
        else:
            spec.append((x, y, s, c))
    return spec

def generate_shape_grid_episode(grammar, min_spec_len=1, max_spec_len=8, held_out={}, max_tries=500):
    tries = 0
    j = 0
    while True:
        programs = grammar.sample(torch.ones((1, 7, 12)))
        programs_list = programs.detach().tolist()
        j += 1
        if grammar.valid(programs_list[0]) and not grammar.hash(programs_list[0]) in held_out:
            spec = sample_S0_shape_grid_spec(grammar, programs_list[0])
            return programs, spec
        elif tries > max_tries:
            print("Max tries exceeded!")
            break

def create_validation_set(grammar, n):
    held_out = set()
    for i in range(n):
        program, _ = generate_shape_grid_episode(grammar, held_out=held_out)
        program_list = program.detach().tolist()
        held_out.add(grammar.hash(program_list[0]))
    return held_out

if __name__ == "__main__":
    grammar = ShapeGridGrammar(7)
    val_programs = create_validation_set(grammar, 100)
    for i in range(50):
        program, spec = generate_shape_grid_episode(grammar, held_out=val_programs)
        print(program.detach().tolist()[0], spec)