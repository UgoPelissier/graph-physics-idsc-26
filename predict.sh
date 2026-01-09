export WANDB_MODE=offline

python -m graphphysics.predict \
            --project_name=idsc_26_predict \
            --predict_parameters_path=predict_config/aneurysm.json \
            --model_path=checkpoints/model.ckpt \
            --prediction_save_path=predictions \
