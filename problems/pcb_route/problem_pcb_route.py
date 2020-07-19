from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.pcb_route.state_pcb_route import StatePcbRoute
import copt
from utils.beam_search import beam_search


class PcbRoute(object):
    NAME = 'PcbRoute'

    @staticmethod
    def get_costs(dataset, pi):
        """
        # Currently, both invalid and valid tour can be evaluated. Later to switch to valid tour by backtracking.
        # Valid or invalid here is determined by seeing whether it can route all the data pairs, but not every node
        # is visited only once.
        """

        penalty = 1e5 * pi.size()[1]
        # dataset: (batch_size, graph_size, node_dim)
        # pi: (batch_size, graph_size)
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"
        # 'out' here is used to make the data format and the data type of the new tensor align with the old one.

        costs = []
        for problem, order in zip(dataset, pi):
            eval_dict = copt.evaluate(list(map(tuple, problem.type(torch.long).tolist())), order.type(torch.long).tolist())
            if not eval_dict['success']:
                costs.append(penalty)
            else:
                costs.append(eval_dict['measure'])

        return torch.tensor(costs), None
        # return (batch_size,), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PcbRouteDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePcbRoute.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        raise NotImplementedError
        # assert model is not None, "Provide model"
        #
        # fixed = model.precompute_fixed(input)

        # def propose_expansions(beam):
        #     return model.propose_expansions(
        #         beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
        #     )
        #
        # state = PcbRoute.make_state(
        #     input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        # )
        #
        # return beam_search(state, beam_size, propose_expansions)


class PcbRouteDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        # size: graph_size
        # num_samples: val_size
        super(PcbRouteDataset, self).__init__()

        self.data_set = []
        # if filename is not None:
        #     assert os.path.splitext(filename)[1] == '.pkl'
        #
        #     with open(filename, 'rb') as f:
        #         data = pickle.load(f)
        #         self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]

        # Generate data before training...
        # Not implemented yet

        # Generate data on the fly...(can not be parallelized)
        self.data = [torch.tensor(copt.getProblem(size), dtype=torch.float) for _ in range(num_samples)]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
