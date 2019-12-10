GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_nips myLMDB/ \
	--train_cvpr myLMDB/ \
	--valroot myLMDB/ \
	--workers 4 \
	--batchSize 20 \
	--niter 1000 \
	--lr 0.25 \
	--cuda \
	--experiment output/myexp \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder
