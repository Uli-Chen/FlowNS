import argparse
from trainer import Trainer
from rich.console import Console


def main():
    print(">>> Starting Main Script Pipeline...")
    parser = argparse.ArgumentParser(description="Flow Matching Negative Sampling Pipeline")
    parser.add_argument('--data_dir', type=str, default='data/MINDsmall_train', help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/MINDsmall_dev/MINDsmall_dev', help='Path to validation dataset')
    parser.add_argument('--dataset', type=str, default='mind', choices=['mind', 'ml100k'], help='Dataset type')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Epochs to pretrain MF')
    parser.add_argument('--fm_epochs', type=int, default=3, help='Epochs to train FM generator')
    parser.add_argument('--finetune_epochs', type=int, default=3, help='Epochs to finetune MF with FM negatives')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of eval cycles)')
    parser.add_argument('--stage1_only', action='store_true', help='Run only Stage 1 (MF pretrain)')
    parser.add_argument('--wandb_project', type=str, default='FlowNS', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity/team')
    parser.add_argument('--wandb_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='offline',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')

    args = parser.parse_args()

    run_config = {
        'data_dir': args.data_dir,
        'val_dir': args.val_dir,
        'dataset': args.dataset,
        'dim': args.dim,
        'batch_size': args.batch_size,
        'pretrain_epochs': args.pretrain_epochs,
        'fm_epochs': args.fm_epochs,
        'finetune_epochs': args.finetune_epochs,
        'eval_every': args.eval_every,
        'patience': args.patience,
        'stage1_only': args.stage1_only,
    }
    
    print(f">>> Initializing Trainer with dataset: {args.data_dir} (type={args.dataset})")
    trainer = Trainer(data_dir=args.data_dir,
                      embedding_dim=args.dim,
                      batch_size=args.batch_size,
                      val_dir=args.val_dir,
                      eval_every=args.eval_every,
                      dataset_type=args.dataset,
                      stage1_only=args.stage1_only,
                      patience=args.patience,
                      wandb_project=args.wandb_project,
                      wandb_entity=args.wandb_entity,
                      wandb_name=args.wandb_name,
                      wandb_mode=args.wandb_mode,
                      run_config=run_config)
    
    # Stage 1: Pretrain MF with Random Negative Sampling
    print(">>> Trainer initialization complete. Starting Stage 1...")
    trainer.train_mf_stage1(epochs=args.pretrain_epochs)
    
    if not args.stage1_only:
        # Stage 2: Train Flow Matching Generator on Exposure Data
        trainer.train_fm_stage2(epochs=args.fm_epochs)
        
        # Stage 3: Finetune MF with Flow Matching Negatives
        trainer.finetune_mf_stage3(epochs=args.finetune_epochs)
    
    trainer.close()
    
    console = Console()
    console.print(">>> Done", style="black on green")


if __name__ == '__main__':
    main()
