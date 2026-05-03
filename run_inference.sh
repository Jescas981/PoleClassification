#! /bin/bash
python src/run.py \
  --model_path checkpoints/dgcnn_fold_2.pth \
  --input_folder ply_folder \
  --output_file labels.txt \
  --num_points 312