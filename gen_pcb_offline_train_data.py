import json
import copt


def generate_offline_pcb_data(n_lines=5, size=50000, early_exit=0, path='pcb_5_5k_bruteforce_data'):
    result = []
    failed_generation = 0
    for _ in range(size):
        problem = copt.getProblem(n_lines)
        solutions = copt.bruteForce(problem, early_exit)
        if solutions[0]['success'] == 1:
            result.append((problem, solutions[0]))
        else:
            failed_generation += 1
            print('Generation failed. Total fail: {}.\n'.format(failed_generation))
    with open(path, 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    generate_offline_pcb_data()
