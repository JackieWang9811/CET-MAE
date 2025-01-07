import os
import argparse
from utils import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, \
    BartForSequenceClassification, BertTokenizer, BertConfig, \
    T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config
from dataset import EEG_dataset_add_sentence_mae as EEG_dataset
# from model_mae import CETMAE_project_late
from model_mae_bart import CETMAE_project_late_bart
from optim_new import *
from contrastive_eeg_pretraining.pre_encoder import cscl_model_cet_mae

import math
"""
task1+task2+taskNRv2 : train_sample : 670*16 = 10710
"""


checkpoint_best = './checkpoints/decoding/best/'

if not os.path.exists(checkpoint_best):
    os.makedirs(checkpoint_best)

checkpoint_last = './checkpoints/decoding/last/'
if not os.path.exists(checkpoint_last):
    os.makedirs(checkpoint_last)


def train_mae(train_dataloader, valid_dataloader, model, optimizer, scheduler,tokenizer,
               early_stopping,num_epochs, checkpoint_path,checkpoint_name):

    best_loss = 100000000000
    best_mae_loss = 100000000000
    best_mlm_loss = 100000000000
    best_c_loss = 100000000000
    best_sim_loss = 100000000000
    eeg_mask_ratio = 0.75
    logger.info("eeg_mask_ratio is {}".format(eeg_mask_ratio))
    text_mask_ratio = 0.75
    logger.info("text_mask_ratio is {}".format(text_mask_ratio))
    mlm_loss_weight = 0.1
    logger.info("mlm_loss_weight is {}".format(mlm_loss_weight))
    mae_loss_weight = 1.0
    logger.info("mae_loss_weight is {}".format(mae_loss_weight))
    contrast_loss_weight = 0.01
    logger.info("contrast_loss_weight is {}".format(contrast_loss_weight))
    # best_model_wts = copy.deepcopy(model.state_dict())
    # checkpoint_eeg_encoder= "model_cet_mae_eeg_encoder_mask_25.pt"
    for epoch_idx in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch_idx, num_epochs - 1))
        logger.info('Epoch {}/{}\n'.format(epoch_idx, num_epochs - 1))
        print('-' * 10)
        model.train()
        train_loss = 0
        train_mae = 0
        train_mlm = 0
        train_c = 0
        train_sim = 0
        t0 = time.time()
        logger.info("Epoch {}".format(epoch_idx))
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            input_embeddings, non_normalized_input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask,target_tokenized,text = batch

            """
            ############################# text ################################
            """
            target_tokenized = target_tokenized.to(device)
            target_tokenized['input_ids'] = target_tokenized['input_ids'].squeeze()
            target_tokenized['attention_mask'] = target_tokenized['attention_mask'].squeeze()
            # 
            attention_mask = target_tokenized['attention_mask']
            attention_mask_invert = attention_mask ^ 1
            target_tokenized['attention_mask_invert'] = attention_mask_invert
            ########################## EEG #######################################
            inputs_embeds_batch = input_embeddings.to(device).float()
            # inputs_attn_mask_batch = {Tensor:(32,58)} ,仅由 0 1 组成
            inputs_attn_mask_batch = input_attn_mask.to(device)
            # inputs_attn_mask_invert_bacth= {Tensor:(32,58)}
            inputs_attn_mask_invert_bacth = input_attn_mask_invert.to(device)
            target_input_ids_batch = target_ids.to(device)
            """replace padding ids in target_ids with -100"""
            target_input_ids_batch[target_input_ids_batch == tokenizer.pad_token_id] = -100


            loss_mae, loss_mlm, loss_c, loss_sim, all_loss = model(inputs_embeds_batch,inputs_attn_mask_batch,inputs_attn_mask_invert_bacth,
                                                                   target_tokenized, mask_ratio_e=eeg_mask_ratio,mlm_probability=text_mask_ratio, mlm_loss_weight=mlm_loss_weight, mae_loss_weight=mae_loss_weight, contrast_loss_weight=contrast_loss_weight )

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            train_loss += all_loss.item()
            train_mae +=loss_mae.item()
            train_mlm += loss_mlm.item()
            train_c += loss_c.item()
            train_sim += loss_sim.item()

        # TODO lr_scheduler
        scheduler.step()

        train_epoch_loss = train_loss/len(train_dataloader)  # TODO make sure of this
        train_epoch_mae_loss = train_mae/len(train_dataloader)
        logger.info("Epoch {} train_epoch_mae_loss: {}".format(epoch_idx, train_epoch_mae_loss))
        train_epoch_mlm_loss = train_mlm / len(train_dataloader)
        train_epoch_c_loss = train_c / len(train_dataloader)
        # print("train_epoch_c_loss: ", train_epoch_c_loss)
        logger.info("Epoch {} train_epoch_c_loss: {}".format(epoch_idx, train_epoch_c_loss))
        train_epoch_sim_loss = train_sim / len(train_dataloader)
        print('Epoch {} Train Loss: {:.4f}'.format(epoch_idx, train_epoch_loss))
        logger.info("Epoch {} Train Loss: {:.4f}".format(epoch_idx, train_epoch_loss))
        all_train_losses.append(train_epoch_loss)
        all_train_mae_losses.append(train_epoch_mae_loss)
        all_train_mlm_losses.append(train_epoch_mlm_loss)
        all_train_c_losses.append(train_epoch_c_loss)
        all_train_sim_losses.append(train_epoch_sim_loss)

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_mae = 0
            valid_mlm = 0
            valid_c = 0
            valid_sim = 0
            for batch_idx, batch in enumerate(tqdm(valid_dataloader)):
                input_embeddings, non_normalized_input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, target_tokenized, text = batch

                """
                ############################# text ################################
                """
                target_tokenized = target_tokenized.to(device)
                target_tokenized['input_ids'] = target_tokenized['input_ids'].squeeze()
                target_tokenized['attention_mask'] = target_tokenized['attention_mask'].squeeze()
                attention_mask = target_tokenized['attention_mask']
                attention_mask_invert = attention_mask ^ 1
                target_tokenized['attention_mask_invert'] = attention_mask_invert

                ########################## EEG #######################################
                inputs_embeds_batch = input_embeddings.to(device).float()
                # inputs_attn_mask_batch = {Tensor:(32,58)} ,仅由 0 1 组成
                inputs_attn_mask_batch = input_attn_mask.to(device)
                # inputs_attn_mask_invert_bacth= {Tensor:(32,58)}
                inputs_attn_mask_invert_bacth = input_attn_mask_invert.to(device)
                target_input_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_input_ids_batch[target_input_ids_batch == tokenizer.pad_token_id] = -100


                loss_mae, loss_mlm, loss_c, loss_sim, all_loss = model(inputs_embeds_batch, inputs_attn_mask_batch, inputs_attn_mask_invert_bacth,
                                                                       target_tokenized, mask_ratio_e=eeg_mask_ratio, mlm_probability=text_mask_ratio, mlm_loss_weight=mlm_loss_weight, mae_loss_weight=mae_loss_weight, contrast_loss_weight=contrast_loss_weight)

                valid_loss += all_loss.item()
                valid_mae += loss_mae.item()
                valid_mlm += loss_mlm.item()
                valid_c += loss_c.item()
                valid_sim += loss_sim.item()
            valid_epoch_loss = valid_loss/len(valid_dataloader)
            valid_epoch_mae_loss = valid_mae/len(valid_dataloader)
            logger.info("Epoch {} valid_epoch_mae_loss: {}".format(epoch_idx, valid_epoch_mae_loss))
            valid_epoch_mlm_loss = valid_mlm / len(valid_dataloader)
            valid_epoch_c_loss = valid_c / len(valid_dataloader)
            # print("valid_epoch_c_loss: ",valid_epoch_c_loss)
            logger.info("Epoch {} valid_epoch_c_loss: {}".format(epoch_idx,valid_epoch_c_loss))
            valid_epoch_sim_loss = valid_sim / len(valid_dataloader)

            print('Epoch {} Valid Loss: {:.4f}'.format(epoch_idx, valid_epoch_loss))
            logger.info("Epoch {} Valid Loss: {:.4f}".format(epoch_idx, valid_epoch_loss))
            all_valid_losses.append(valid_epoch_loss)
            all_valid_mae_losses.append(valid_epoch_mae_loss)
            all_valid_mlm_losses.append(valid_epoch_mlm_loss)
            all_valid_c_losses.append(valid_epoch_c_loss)
            all_valid_sim_losses.append(valid_epoch_sim_loss)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            best_epoch = epoch_idx
            saved_name = os.path.join(checkpoint_path,checkpoint_name)
            torch.save(model.state_dict(), saved_name)

            print("save the best cet-mae checkpoint")
            logger.info("save the best cet-mae checkpoint")

            # encoder_saved_name = os.path.join(checkpoint_path, checkpoint_eeg_encoder)
            # desired_state_dict = {key: value for key, value in model.state_dict().items() if key in desired_weights}
            # # 保存特定权重
            # torch.save(desired_state_dict, encoder_saved_name)

            # best_model_wts = copy.deepcopy(model.state_dict())
        if valid_epoch_mae_loss< best_mae_loss:
            best_mae_loss = valid_epoch_mae_loss
            best_mae_epoch = epoch_idx

        if valid_epoch_mlm_loss< best_mlm_loss :
            best_mlm_loss = valid_epoch_mlm_loss
            best_mlm_epoch = epoch_idx

        if valid_epoch_c_loss< best_c_loss:
            best_c_loss = valid_epoch_c_loss
            best_c_epoch = epoch_idx

        if valid_epoch_sim_loss < best_sim_loss:
            best_sim_loss = valid_epoch_sim_loss
            best_sim_epoch = epoch_idx

        # if early_stopping.early_stop(valid_epoch_loss):
        #     print("We are at epoch:", epoch_idx)
        #     break

    print("best_epoch:",best_epoch)
    logger.info("best_epoch:{}".format(best_epoch))
    print("best_mae_epoch:",best_mae_epoch)
    logger.info("best_mae_epoch:{}".format(best_mae_epoch))
    print("best_mlm_epoch:",best_mlm_epoch)
    logger.info("best_mlm_epoch:{}".format(best_mlm_epoch))
    print("best_contrastive_epoch:",best_c_epoch)
    logger.info("best_contrastive_epoch:{}".format(best_c_epoch))
    # print("best_sim_epoch:",best_sim_epoch)
    # logger.info("best_sim_epoch:{}".format(best_sim_epoch))


    final_epoch_save_name = os.path.join(checkpoint_path, f"{checkpoint_name}_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_epoch_save_name)
    print(f"Saved final epoch weights to {final_epoch_save_name}")
    logger.info(f"Saved final epoch weights to {final_epoch_save_name}")


    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

def print_and_save_model_structure(model, filepath):
    with open(filepath, 'w') as f:
        print('Model Structure:', file=f)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(' ', name, file=f)

def plot_loss_trend(train_losses, valid_losses, save_path, save_title):
    # Plotting the training and validation loss
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='*')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(save_title)
    plt.legend()
    plt.xticks(epochs[::5])  


    # for i, loss in enumerate(train_losses):
    #     plt.text(epochs[i], loss, f'{loss:.4f}', ha='right', va='bottom',fontsize=6)  # 在训练损失数据点上标注数值
    #
    # for i, loss in enumerate(valid_losses):
    #     plt.text(epochs[i], loss, f'{loss:.4f}', ha='right', va='bottom',fontsize=6)  # 在验证损失数据点上标注数值

    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to release memory


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EEG-Text')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    args = vars(parser.parse_args())
    args = read_configuration(args["config"])

    ''' config param'''
    num_epoch_mae = args['num_epoch_pretrain']

    # num_epoch_fintune = args['num_epoch_fintune']
    # lr_clip = args['lr_clip']
    # lr_fintune = args['lr_fintune']

    batch_size = args['batch_size']

    model_name = args['model_name']

    init_logger(args)
    logger = getLogger()

    save_path = args['save_path']


    print(f'[INFO]using model: {model_name}')

    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    """save config"""
    tokenizer = BartTokenizer.from_pretrained('./models/huggingface/bart-large')




    """ dataset """
    train_set = EEG_dataset(path=args["dataset_path"] + "train")
    valid_set = EEG_dataset(path=args["dataset_path"] + "valid")
    test_set = EEG_dataset(path=args["dataset_path"]+ "test")

    train_set = ConcatDataset([train_set,valid_set])
    # dataset_sizes = {'train': len(train_set), 'dev': len(valid_set),'test':len(test_set)}
    dataset_sizes = {'train': len(train_set), 'test': len(test_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]test_set size: ', len(test_set))

    """ dataloader """
    train_dataloader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=4)
    # valid_dataloader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=4)

    # dataloaders = {'train': train_dataloader, 'dev': valid_dataloader,'test':test_dataloader}
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}

    model = CETMAE_project_late_bart(multi_heads=8,feedforward_dim=2048,trans_layers=6, device=device)
    logger.info("the model is CETMAE_project_late_bart")

    model.to(device)

    # parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args, model, mode="cet-mae")

    early_stopping = EarlyStopper(patience=4, min_delta=0.005)
    # optimizer = torch.optim.AdamW(parameters, args["cet_mae_lr"], weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    # if args["lr_adapt"] == True:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    #     print('Override to use adaptive learning rate scheduler.')
    # else:
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args["lrscheduler_start"], 1000, args["lrscheduler_step"])),gamma=args["lrscheduler_decay"])
    #     print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args["lrscheduler_start"], args["lrscheduler_decay"], args["lrscheduler_step"]))

    print('=== start Step2 training ... ===')
    # print training layers
    print_and_save_model_structure(model,'./model_struct.txt')
    all_train_losses = []
    all_train_mae_losses = []
    all_train_mlm_losses = []
    all_train_c_losses = []
    all_train_sim_losses = []

    all_valid_losses = []
    all_valid_mae_losses = []
    all_valid_mlm_losses = []
    all_valid_c_losses = []
    all_valid_sim_losses = []

    logger.info(args['cet_mae_lr'])

    print("save_path: ", args["save_path"])
    print("cet_mae_checkpoint_name: ", args["cet_mae_checkpoint_name"])
    logger.info(args["cet_mae_checkpoint_name"])
    # print("checkpoint_eeg_encoder: ", args['checkpoint_eeg_encoder'])
    # logger.info(args['checkpoint_eeg_encoder'])

    train_mae(train_dataloader, test_dataloader, model, optimizer, scheduler,tokenizer,early_stopping,
              num_epochs=num_epoch_mae,
              checkpoint_path=args["save_path"],
              checkpoint_name = args["cet_mae_checkpoint_name"])

    folder_name = args['folder_name']
    directory = f'./loss_plot/{folder_name}'

    # 
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_loss_trend(all_train_losses, all_valid_losses, f'{directory}/cet_mae_pretrain_total_loss_project_late.png','CET-MAE Total Loss')
    plot_loss_trend(all_train_mae_losses, all_valid_mae_losses, f'{directory}/cet_mae_pretrain_mae_loss_project_late.png','CET-MAE EEG-MAE Loss')
    plot_loss_trend(all_train_mlm_losses, all_valid_mlm_losses, f'{directory}/cet_mae_pretrain_mlm_loss_project_late.png','CET-MAE Text-MLM Loss')
    plot_loss_trend(all_train_c_losses, all_valid_c_losses, f'{directory}/cet_mae_pretrain_contrastive_loss_project_late.png','CET-MAE EEG-Text Contrastive Loss')
    plot_loss_trend(all_train_sim_losses, all_valid_sim_losses,f'{directory}/cet_mae_pretrain_similarity_loss_project_late.png','CET-MAE EEG Similarity Loss')
