import os
import json
import math
import torch
import argparse
import logging

from datetime import timedelta
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from transformers import (
    AdamW,   
    SchedulerType,
    get_scheduler,
    set_seed,
)

from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
from accelerate import (
    Accelerator, 
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)

from processors.dataloaders import R2DataLoader
from eval_cxr.metrics import compute_scores, load_gts

logger = logging.getLogger(__name__)


def postprocess_text(preds, labels):
    preds = [pred.strip().replace("\n", "") for pred in preds]
    labels = [label.strip().replace("\n", "") for label in labels]

    # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # path to the directories containing datasets and models 
    parser.add_argument('--image_dir', type=str, default='./idl/mimic-cxr',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='./idl/annts',
                        help='the path to the directory containing the data.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to tokenizer from huggingface.co/models.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    
    # Model args
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--valid_steps",
        type=int,
        default=None,
        help="How frequently validate the checkpoint during the training phase",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=0, help="Number of steps for logging current loss."
    )

    # text generation
    parser.add_argument(
        "--beam_size", type=int, default=5, help=""
    )
    parser.add_argument(
        "--min_len", type=int, default=0, help=""
    )
    parser.add_argument(
        "--max_len", type=int, default=50, help=""
    )
    parser.add_argument(
        "--max_len_b", type=int, default=0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--length_penalty", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help=""
    )

    # wnadb
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default='test', help=""
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default='test', help=""
    )
        
    # etc.
    parser.add_argument(
        "--do_train", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--do_valid", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--do_eval", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--do_not_save_models", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--custom_vocab", action="store_true", default=False, help=""
    )

    args = parser.parse_args()
    
    return args

def evaluate(args,
            model,
            tokenizer,
            eval_dataloader, 
            inputs_prompt,
            accelerator,
            phase="test"
            ):
        
        logger.info(f"***** Running evaluation for {phase}*****")
        logger.info(f"  Num steps = {len(eval_dataloader)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

        generator = sequence_generator.SequenceGenerator(
                                            tokenizer=tokenizer,
                                            beam_size=args.beam_size,
                                            max_len_b=args.max_len,
                                            min_len=args.min_len,
                                            temperature=args.temperature,
                                            no_repeat_ngram_size=3,
                                            len_penalty=1.0,
                                            )
        
        all_preds = []
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                input_ids = inputs_prompt.expand([len(batch[1]), -1]) # expand the text prompt to the batch size
                
                net_input = {"input_ids" : input_ids,
                             "patch_images" : batch[1],
                             "patch_masks"  : torch.tensor([True]),
                            }
                
                net_input = {key: inp.to(accelerator.state.device) for key, inp in net_input.items()}
                inputs = {"net_input" : net_input
                        }

                gen_output = generator.generate([accelerator.unwrap_model(model)], inputs)
                gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
                
                """ using huggingface's module -> it takes longer
                gen = model.generate(**net_input,
                                    do_sample=True,
                                    no_repeat_ngram_size=3,
                                    min_length=args.min_len,
                                    max_length=args.max_len,
                                    num_beams=args.beam_size,
                                    temperature=args.temperature,
                                    length_penalty=args.length_penalty,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    )
                """
                preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
                
                all_preds += [pred.strip() for pred in preds]
                progress_bar.update(1)
        if accelerator.is_main_process:
            all_gts = load_gts(args.ann_path, split='test')

            all_preds, all_gts = postprocess_text(all_preds, all_gts)

            scores = compute_scores({i: [gt] for i, gt in enumerate(all_gts)},
                                    {i: [re] for i, re in enumerate(all_preds)}
                                    )
            
            return all_preds, scores
        else:
            return None, None
        

def main():
    
    args = parse_arguments()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=10800))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])
    logger.info(accelerator.state)
    logger.info(accelerator.state.device)
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
    
    ## logging with wandb 
    if args.use_wandb and accelerator.is_main_process:
        import wandb
        ## wandb setup
        os.environ["WANDB_API_KEY"]=''
        os.environ["WANDB_PROJECT"]=args.wandb_project_name
        os.environ["WANDB_WATCH"]='false'
        os.environ["WANDB_START_METHOD"]='thread'
        os.environ["WANDB_USER_EMAIL"]='jh.wlghlee@gmail.com'
        os.environ["WANDB_USERNAME"]='JihoLee'

        wandb.init()
        wandb.config.update(args)
        wandb.run.name = args.wandb_run_name
    

    ## define a technique for transforming image into features
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])
    
    
    # Load tokenizer and model
    logger.info("Load the tokenizer and model")
    tokenizer = OFATokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path is not None else args.model_name_or_path,
                                             use_fast=True)
    pad_token_id = tokenizer.encode(str(tokenizer._pad_token), add_special_tokens=False)[0]
    model = OFAModel.from_pretrained(args.model_name_or_path, use_cache=False)

    if args.custom_vocab:
        model.resize_token_embeddings(len(tokenizer))

    # Load the dataloadenr
    logger.info("Load the dataloaders")
        
    with accelerator.main_process_first():
        if args.do_train:
            train_dataloader = R2DataLoader(args, tokenizer, 
                                            split='train', 
                                            shuffle=True, 
                                            batch_size=args.per_device_train_batch_size,
                                            transform=patch_resize_transform, # r2gen_transform
                                )

        valid_dataloader = R2DataLoader(args, tokenizer, 
                                        split='val', 
                                        shuffle=False, 
                                        batch_size=args.per_device_eval_batch_size,
                                        transform=patch_resize_transform, # r2gen_transform
                            )
        
        test_dataloader = R2DataLoader(args, tokenizer, 
                                        split='test', 
                                        shuffle=False, 
                                        batch_size=args.per_device_eval_batch_size,
                                        transform=patch_resize_transform, # r2gen_transform
                            )
    
    # define text prompt depending on the target task
    txt_prompt = " what does the image describe?"
    inputs_prompt = tokenizer([txt_prompt], return_tensors="pt").input_ids
    
    if args.do_train:
        # def train():
        
        ## Load the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # TODO: inference with multiple GPUs
        # Prepare everything with our `accelerator`.
        # if args.do_train:
        #     model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        #         model, optimizer, train_dataloader, valid_dataloader, test_dataloader
        #     )
        # else:
        #     model, optimizer, test_dataloader = accelerator.prepare(
        #         model, optimizer, test_dataloader
        #     )
        # Evaluating with single gpu 
        model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader
        )

        
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        
        ## Load LR scheduler
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        loss_fct = CrossEntropyLoss(ignore_index=pad_token_id,
                                    label_smoothing=args.label_smoothing)
    
        # Train!
        total_batch_size = args.per_device_train_batch_size # * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps)) #, disable=not accelerator.is_local_main_process)
        completed_steps = 0
        best_bleu4 = 0.0
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                # outputs = model(**batch)
                # loss = outputs.loss
                
                input_ids = inputs_prompt.expand([len(batch[1]), -1]) # expand the text prompt to the batch size
                
                inputs = {"net_input" : {"input_ids" : input_ids,
                                        "patch_images" : batch[1],
                                        "patch_masks"  : torch.tensor([True]),
                                        "labels" : batch[2],
                                        "attention_mask" : batch[3],
                                        }
                        }
                
                net_input = {key: inp.to(accelerator.state.device) for key, inp in inputs["net_input"].items()}
                outputs = model(**net_input)
                
                lm_logits = outputs[0]
                labels    = net_input["labels"]
                
                loss = loss_fct(lm_logits.view(-1, len(tokenizer)), labels.view(-1))
                loss /= args.gradient_accumulation_steps
                
                # loss.backward()
                accelerator.backward(loss)
                
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break
                
                if (step + 1) % args.logging_steps == 0 and accelerator.is_main_process:
                    logger.info(f"Loss : {loss.item()}")
                    # logger.info(f"{step} / {len(train_dataloader)}")
                    if args.use_wandb:
                        wandb.log({"train loss" : loss.item(),
                                  "step"       : completed_steps,
                                  "epoch"       : epoch,
                                  })
                
                # Validate and save!
                if args.do_valid and (completed_steps + 1) % args.valid_steps == 0:
                    model.eval()
                    if accelerator.is_main_process:
                        all_preds, scores = evaluate(args,
                                                    model,
                                                    tokenizer,
                                                    valid_dataloader,
                                                    inputs_prompt,
                                                    accelerator,
                                                    phase="valid"
                                                    )
                    
                        bleu4 = scores["BLEU_4"]
                        scores = {'val_' + k: v for k,v in scores.items()}
                        logger.info(f"{completed_steps} step has ended")
                        logger.info(scores)

                        if bleu4 > best_bleu4:
                            logger.info("best checkpoint has been changed")
                            # ckpt_dir = os.path.join(args.output_dir, f'ckpt_{epoch}_' + str(step + 1))
                            os.makedirs(args.output_dir, exist_ok=True)
                            
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(args.output_dir)
                            
                            pred_path = os.path.join(args.output_dir, f'preds.json')
                            
                            with open(pred_path, 'w') as fp:
                                json.dump(all_preds, fp)
                            
                            with open(os.path.join(args.output_dir, 'scores.txt'), 'a') as f:
                                f.write(f'ckpt_{epoch}_' + str(step + 1) + ": " + str(scores))
                        
                            if args.use_wandb:
                                wandb.log(scores)
                    
                    accelerator.wait_for_everyone()

                    model.train()
                    
        logger.info(f"Training has ended.")
    
    # Evaluate!
    if args.do_eval:
        if args.do_train:
            logger.info("Load the tokenizer and model")
            tokenizer = OFATokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path is not None else args.output_dir,
                                                    use_fast=True)
            model = OFAModel.from_pretrained(args.output_dir, use_cache=False)
            model = accelerator.prepare(model)

        model.eval()
        if accelerator.is_main_process:
            all_preds, scores = evaluate(args,
                                        model,
                                        tokenizer,
                                        test_dataloader, 
                                        inputs_prompt,
                                        accelerator,
                                        phase="test"
                                        )

            logger.info(f"Save the final model into {args.output_dir}")
            logger.info("Final results are as follows")
            logger.info(scores)
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, 'preds.json')
            
            if not args.do_not_save_models:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

            with open(out_path, 'w') as fp:
                json.dump(all_preds, fp)
                
            with open(os.path.join(args.output_dir, 'scores.txt'), 'a') as f:
                f.write(str(scores))

            if args.use_wandb:
                wandb.log(scores)
    
    accelerator.wait_for_everyone()
    
if __name__ == "__main__":
    main()
