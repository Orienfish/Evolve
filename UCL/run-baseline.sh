# Usage:
#   Run UCL on various dataset under four streaming settings: iid, seq, seq-bl, seq-cc, e.g.,
#     ./run-baseline.sh mixup supcon mnist iid trial#
#   Method choices: mixup, pnn, si, der
#   Model name choices: supcon, simsiam, barlowtwins
#   Dataset choices: mnist, svhn, cifar10, cifar100, tinyimagenet
#   Datatype choices: iid, seq, seq-bl, seq-cc, seq-im
#   Trial #: the number of trial

model=resnet18
lr=0.03;

if [ "$3" = "mnist" ] || [ "$3" = "svhn" ]; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type iid  \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone cnn --training_data_type class_iid --imbalanced \
      --batch_size 256 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "cifar10" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid\
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --imbalanced \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "cifar100" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid  \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --val_freq 20 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --imbalanced \
      --batch_size 128 --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "tinyimagenet" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid  \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-bl" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-cc" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_iid --imbalanced \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "core50" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid  \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.2 --val_samples_ratio 0.1 --knn_samples_ratio 0.2 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_instance \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.2 --val_samples_ratio 0.1 --knn_samples_ratio 0.2 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_instance --imbalanced \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.2 --val_samples_ratio 0.1 --knn_samples_ratio 0.2 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

if [ "$3" = "stream51" ] ; then
  if [ "$4" = "iid" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type iid  \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_instance \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi

  if [ "$4" = "seq-im" ]; then
    python main.py --method "$1" --model_name "$2" --dataset "$3" --backbone $model --training_data_type class_instance --imbalanced \
      --batch_size 128 --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial "$5" \
      -c configs/"$3".yaml --log_dir ../logs/ --ckpt_dir ./checkpoints/"$3"_results/
  fi
fi

