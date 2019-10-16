BATCH=5
TEST_BATCH=5
VOCAB=500
MLEN=200
DATA='music'
FIXED_EPOCHS=10
NUM_EPOCHS=50
NHIDDEN=128
PRETRAINED=""
LAST_EPOCH=9
SAMPLE_TIME=3
LEARNING_RATE=0.05
GRAD_CLIP=1
BATCH_NORM=1
SIGMOID=1

FLAGS="floatX=float32,device=cuda"
THEANO_FLAGS="${FLAGS}" python main.py --model TLSTM3 --data ${DATA} \
    --batch_size ${BATCH} --vocab_size ${VOCAB} --max_len ${MLEN} \
    --fixed_epochs ${FIXED_EPOCHS} --num_epochs ${NUM_EPOCHS} \
    --num_hidden ${NHIDDEN} --test_batch ${TEST_BATCH} \
    --learning_rate ${LEARNING_RATE} --sample_time ${SAMPLE_TIME} \
    --grad_clip ${GRAD_CLIP} --bn ${BATCH_NORM} --sigmoid_on ${SIGMOID}

