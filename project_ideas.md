#### easiest training demo (copy and run in terminal)

    python run.py --problem PcbRoute --graph_size 10 --baseline rollout --batch_size 32 --epoch_size 128 --val_size 16 --embedding_dim 64 --hidden_dim 64 --n_epochs 20 --eval_batch_size 4 --run_name 'PcbRoute10_rollout'

#### to run tensorboard UI
	tensorboard --logdir logs/tsp_10 
(replace tsp_10 with the directory name you want to view)

#### ideas on experiments:
Original paper:

use various problems (not applicable here), various problem size(\*) and various model. Various models have two parts: one is using different baselines(exponential, rollout, critic), the other is using different models(PN, AM, OR tools etc.). Comparison between different models with different baselines. Perhaps different training method. Also, different docoding strategy. Greedy decoding models are compared together while sampling decoding models are in another group to compare. Best possible solution for each model in 10000 test instances. For speed, there are way too many values, but we should only pick up the most valuable parts.
Tables for hyper-parameters comparison, while figures for model comparison.

#### Difference between classic combinatorial optimization problems and the pcb multiple routing problem

The classic combinatorial problems have limited constraints. In TSP, the only constraint is each node can be visited only once. This can be done via masking process. However, other constraints may not be solves easily by masking. All kinds of constraints all transferred into masks. This applies to pcb routing problem.


#### TODO (validate):
- While validating model, document the optimal value using brute-force method.
Compare the validation output against optimal value.
(Make sure sampling rather than greedy decoder is implemented while validating)
(Beam search hasn't been implemented yet)

- validate on different graph sizes (5, 10, 20, 50, 100)

Then validate on different decoder.

#### TODO (decoder):
Design three decoders: 
- One using masking. masking out all the nodes that can not pass the eval function of copt. (This is too computational expensive.)
- One using backtracking, re-sample the illegal nodes until satisfied.
- One using penalizing.

Compare their performances in experiment.

The third one is the easiest, should be done before 20th July. (Done)

	python run.py --problem PcbRoute --graph_size 5 --baseline rollout --batch_size 32 --epoch_size 8192 --val_size 128 --embedding_dim 128 --hidden_dim 128 --n_epochs 20 --eval_batch_size 32 --run_name 'PcbRoute5_rollout'
当前的进展：

惩罚reward作为decoder已完成；

目前的问题：

之后的规划：

准备做回溯法decoder