from trainer.trainer_framework import BaseTrainer
from config.config import parse_arguments_and_create_config
from utils.logger import initialize_wandb

def main():
    # Get configurations and initialize
    configs = parse_arguments_and_create_config()
    logger = initialize_wandb(configs)

    # Initialize and run trainer
    trainer = BaseTrainer(
        trainer_config=configs["trainer_config"],
        dataset_config=configs["dataset_config"],
        model_config=configs["model_config"], 
        bool_config=configs["boolean_config"],
        feature_selector_config=configs["feature_selector_config"],
        logger=logger
    )

    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Run training if not just extracting masks
    if not (configs["trainer_config"].extract_masks and configs["trainer_config"].skip_training):
        results = trainer.train()
    
    # Extract masks if requested and using SFW approach
    if configs["trainer_config"].extract_masks:
        if configs["trainer_config"].approach in ["sfw", "nn"]:
            print("\nExtracting and saving feature masks from best model...")
            masks_path = trainer.extract_and_save_masks(
                configs["trainer_config"].masks_output_path,
                configs["trainer_config"].checkpoint_path
            )
            print(f"Masks saved to: {masks_path}")
        else:
            print("\nMask extraction is only supported for the SFW approach.")
    
if __name__ == "__main__":
    main()