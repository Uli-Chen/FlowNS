import argparse
from trainer import Trainer

def main():
    print(">>> Starting Main Script Pipeline...")
    parser = argparse.ArgumentParser(description="Flow Matching Negative Sampling Pipeline")
    parser.add_argument('--data_dir', type=str, default='data/MINDsmall_train', help='Path to dataset')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--pretrain_epochs', type=int, default=3, help='Epochs to pretrain MF')
    parser.add_argument('--fm_epochs', type=int, default=3, help='Epochs to train FM generator')
    parser.add_argument('--finetune_epochs', type=int, default=3, help='Epochs to finetune MF with FM negatives')
    
    args = parser.parse_args()
    
    print(f">>> Initializing Trainer with dataset: {args.data_dir}")
    trainer = Trainer(data_dir=args.data_dir, 
                      embedding_dim=args.dim, 
                      batch_size=args.batch_size)
    
    # Stage 1: Pretrain MF with Random Negative Sampling
    print(">>> Trainer initialization complete. Starting Stage 1...")
    trainer.train_mf_stage1(epochs=args.pretrain_epochs)
    
    # Stage 2: Train Flow Matching Generator on Exposure Data
    trainer.train_fm_stage2(epochs=args.fm_epochs)
    
    # Stage 3: Finetune MF with Flow Matching Negatives
    trainer.finetune_mf_stage3(epochs=args.finetune_epochs)
    
    print("Pipeline Execution Complete!")

if __name__ == '__main__':
    main()
