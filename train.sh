export WANDB_MODE=offline

python -m graphphysics.train \
            --project_name=fixed-2d-stent-vitesse-cup-cap-dbp \
            --training_parameters_path=training_config/aneurysm.json \
            --num_epochs=100 \
            --init_lr=0.001 \
            --batch_size=1 \
            --warmup=1000 \
            --num_workers=0 \
            --prefetch_factor=0 \
            --model_save_name=model \
            --use_previous_data \
            --previous_data_start 5 \
            --previous_data_end 10 \