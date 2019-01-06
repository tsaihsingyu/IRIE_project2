environment: python3
package: keras, tensorflow, sklearn, numpy, gensim

(1) 訓練模型、 產生 node label 預測檔
	python3 predict_node.py --do_train=1 --use_pos=<1 if using pos tag else 0>

(2) 給定 node label，預測 edge label
	a. task 1: node label 是自己產生的
		python3 predict_edge.py --node_file=<path/to/node_*.npy/file>

	b. task 2
		python3 predict_edge.py