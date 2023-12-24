# Usage:
#   Run SimCLR on ALL dataset under FIVE streaming settings: iid, seq, seq-bl, seq-cc, seq-im, e.g.,
#     ./run-baseline.sh simclr mnist iid trial#
#   Criterion choices: simclr, barlowtwins, simsiam, vicreg, byol
#   Dataset choices: mnist, svhn, cifar10, cifar100, tinyimagenet, core50, stream51
#   Data stream choices: iid, seq, seq-bl, seq-cc, seq-im
#   Trial #: the number of trial

cd ..;

model=resnet18;
lr=0.03;
mem_samples=0;  #128;
mem_size=0;  #256;
size=32;

if [ $2 = "mnist" ] || [ $2 = "svhn" ]; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model cnn --training_data_type iid  \
            --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
            --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
            --learning_rate_stream $lr --temp_cont 0.1 \
            --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model cnn --training_data_type class_iid \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 \
        --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model cnn --training_data_type class_iid --blend_ratio 0.5 \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 \
        --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model cnn --training_data_type class_iid --n_concurrent_classes 2 \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 \
        --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model cnn --training_data_type class_iid --imbalanced \
        --batch_size 256 --mem_samples $mem_samples --mem_size $mem_size \
        --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 20 --epochs 1 \
        --learning_rate_stream $lr --temp_cont 0.1 \
        --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi
fi


if [ $2 = "cifar10" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid --imbalanced \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi
fi


if [ $2 = "cifar100" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1  --mem_size_per_class 64 --mem_max_classes 20 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --criterion $1 --dataset $2 --model $model --training_data_type class_iid --imbalanced \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 128 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 1.0 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi
fi


if [ $2 = "tinyimagenet" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --size $size --criterion $1 --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --size $size --criterion $1 --dataset $2 --model $model --training_data_type class_iid \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-bl" ]; then
    python main_supcon.py --size $size --criterion $1 --dataset $2 --model $model --training_data_type class_iid --blend_ratio 0.5 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-cc" ]; then
    python main_supcon.py --size $size --criterion $1 --dataset $2 --model $model --training_data_type class_iid --n_concurrent_classes 2 \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --size $size --criterion $1 --dataset $2 --model $model --training_data_type class_iid --imbalanced \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.5 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 --trial $4
  fi
fi

if [ $2 = "stream51" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --size $size --criterion $1 --lifelong_method none --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples $mem_samples --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 \
      --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --size $size --criterion $1 --lifelong_method none --dataset $2 --model $model --training_data_type class_instance \
      --batch_size 128 --mem_samples 128 --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 \
      --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --size $size --criterion $1 --lifelong_method none --dataset $2 --model $model --training_data_type class_instance --imbalanced \
      --batch_size 128 --mem_samples 128 --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.9 --val_samples_ratio 0.1 --knn_samples_ratio 1.0 \
      --trial $4
  fi
fi

if [ $2 = "core50" ] ; then
  if [ $3 = "iid" ]; then
    python main_supcon.py --size $size --criterion $1 --lifelong_method none --dataset $2 --model $model --training_data_type iid  \
      --batch_size 128 --mem_samples 128 --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.2 --val_samples_ratio 0.1 --knn_samples_ratio 0.2 \
      --trial $4
  fi

  if [ $3 = "seq" ]; then
    python main_supcon.py --size $size --criterion $1 --lifelong_method none --dataset $2 --model $model --training_data_type class_instance \
      --batch_size 128 --mem_samples 128 --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.2 --val_samples_ratio 0.1 --knn_samples_ratio 0.2 \
      --trial $4
  fi

  if [ $3 = "seq-im" ]; then
    python main_supcon.py --size $size --criterion $1 --lifelong_method none --dataset $2 --model $model --training_data_type class_instance --imbalanced \
      --batch_size 128 --mem_samples 128 --mem_size $mem_size \
      --val_batch_size 64 --num_workers 8 --steps_per_batch_stream 10 --print_freq 10 --epochs 1 \
      --learning_rate_stream $lr --temp_cont 0.1 \
      --train_samples_ratio 0.4 --test_samples_ratio 0.2 --val_samples_ratio 0.1 --knn_samples_ratio 0.2 \
      --trial $4
  fi
fi
