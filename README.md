We propose two Hierarchical Transformers for Multi-document summarization, namely Parallel HT and Vertical HT.

Preparation
-------
 You can find the best checkpoints from https://pan.baidu.com/s/18kZpVQuuPpDjcRoQ6D6YnQ, code: uzvc 
 
 The ranked WikiSum dataset from https://github.com/nlpyang/hiersumm
 
 You should dowload above checkpoints and dataset and put them in your project.
 
 Traning
 -----
    python main.py --mode train --model_mode p --ckpt_path ./checkpoints/train_large_p_3d_30 --batch_size 16 --epoch 5 --para_len 100 --para_num 30
   
   This is to train Parallel HT, we use default hyper-parameters of the model defined in the paper.
   
    python main.py --mode train --model_mode v --ckpt_path ./checkpoints/train_large_v_3d_30
    
   This is to train Vertical HT.
   
Validation
----
    python main.py --mode valid --model_mode p --ckpt_path ./checkpoints/train_large_p_3d_30
    
   Validation is to find the best checkpoint, You don't need to do this because we have uploaded the best ckeckpoints.
   
 Generating
 ----
     python main.py --mode generate --model_mode p --ckpt_path ./checkpoints/train_large_p_3d_30 --beam_size 5 --block_n_grams 3 -- block_n_words_before 2
   
   This is to use Parallel HT to generate summaries, the results are saved in a txt file. We use beam search, and two regulariztions to prevent repetitive grams during inference.
     
     
     
     
