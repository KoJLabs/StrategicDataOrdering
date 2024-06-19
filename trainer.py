from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from trl import SFTTrainer


class CustomTrainer(SFTTrainer):
    def __init__(self, *args, order_type=None, **kwargs):
        super().__init__(*args, **kwargs) 
        self.order_type = order_type

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        

        if self.order_type == "loss" or self.order_type == "attention" or self.order_type == "length":
            train_sampler = SequentialSampler(train_dataset)

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
            )
        
        else:
            train_sampler = RandomSampler(train_dataset)

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
            )