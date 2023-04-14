
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
import copy
from model.AudioNet import AudioNet
from attack.FGSM import FGSM
from attack.PGD import PGD
from dataset.Spk251_train import Spk251_train
from dataset.Spk251_test import Spk251_test
from dataset.backdoor_Spk251_train import backdoor_Spk251_train
from dataset.backdoor_Spk251_test import backdoor_Spk251_test

import wandb
from attack.sPGD import sPGD
from attack.ePGD import ePGD
from defense.defense import *
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

starttime = time.time()
time.sleep(2.1) #??2.1s


def parser_args(): 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-defense', default=None)
    parser.add_argument('-defense_param', default=None, nargs='+')

    parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    parser.add_argument('-aug_eps', type=float, default=0.002)

    parser.add_argument('-attacker', type=str, choices=['PGD', 'FGSM','sPGD','ePGD'], default='PGD')
    parser.add_argument('-epsilon', type=float, default=0.002) 
    parser.add_argument('-step_size', type=float, default=0.0004) # recommend: epsilon / 5
    parser.add_argument('-max_iter', type=int, default=10) # PGD-10 default
    parser.add_argument('-num_random_init', type=int, default=0)
    parser.add_argument('-EOT_size', type=int, default=1)
    parser.add_argument('-EOT_batch_size', type=int, default=1)
    
    # using root/Spk251_train as training data
    # using root/Spk251_test as validation data
    parser.add_argument('-root', type=str, default='/mnt/data') # directory where Spk251_train and Spk251_test locates
    parser.add_argument('-num_epoches', type=int, default=150)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-wav_length', type=int, default=80_000)

    parser.add_argument('-ratio', type=float, default=0.5) # adversarial examples ratio
    parser.add_argument('-lr', type=float, default=0.2) # backdoor ratio

    parser.add_argument('-model_ckpt', type=str)
    parser.add_argument('-log', type=str)
    parser.add_argument('-ori_model_ckpt', type=str)
    parser.add_argument('-ori_opt_ckpt', type=str)
    parser.add_argument('-start_epoch', type=int, default=0)

    parser.add_argument('-evaluate_per_epoch', type=int, default=1)
    parser.add_argument('-evaluate_adver', action='store_true', default=False) 

    parser.add_argument('-wandb_start', type=int, default=0) 
    parser.add_argument('-wandb_id', type=str, default='') 
    parser.add_argument('-val', type=int, default=0) 
    
    args = parser.parse_args()
    return args


def validation_attack(args, model, val_data, attacker):
    model.eval()
    val_normal_acc = None
    val_adver_acc = None
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                end='\r')
            if decision == true:
                right_cnt += 1
        val_normal_acc = right_cnt / total_cnt
    print()
    if args.evaluate_adver:
        n_select = len(val_data)
        val_adver_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            adver, success = attacker.attack(origin, true)
            decision, _ = model.make_decision(adver) 
            print((f'[{index}/{n_select}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                end='\r')
            if decision == true: 
                val_adver_cnt += 1 
        val_adver_acc = val_adver_cnt / n_select
        print()
    else:
        val_adver_acc = 0.0
    return val_normal_acc, val_adver_acc
    
def validation_benign(model, val_data):
    model.eval()
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                    end='\r')
            if decision == true: 
                right_cnt += 1 
    return right_cnt / total_cnt

def validation_backdoor(model, val_data):
    model.eval()
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), 
                    end='\r')
            if decision == true: 
                right_cnt += 1 
    return right_cnt / total_cnt


def main(args):
    import wandb
    
    wandb_exper_name = 'speaker-cognition_lr' + str(args.lr) + '_epoch' + str(args.num_epoches)

    if args.wandb_start == 0:
        wandb = None # if do not use wandb,should be set to None
    else:
        # os.environ["WANDB_API_KEY"] = '417379ea7214f7bf59d9e63187d2afbdf53b39fd'
        # os.environ["WANDB_MODE"] = "offline"
        if args.wandb_id:
            wandb.init(id=args.wandb_id, resume='must', name=wandb_exper_name, entity='imomoe', project="FL-Speaker-Backdoor-AudioNet")
        else:    
            wandb.init(name=wandb_exper_name, entity='imomoe', project="FL-Speaker-Backdoor-AudioNet")
        
        wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release
        # print("wandbID is : " + wandb_id)
        # save_wandb_id(wandb_exper_name, wandb_id + "\n", )

    # load model
    # speaker info
    defense_param = parser_defense_param(args.defense, args.defense_param)
    model = AudioNet(args.label_encoder,
                    transform_layer=args.defense, 
                    transform_param=defense_param)
    spk_ids = model.spk_ids
    if args.ori_model_ckpt:
        print(args.ori_model_ckpt)
        # state_dict = torch.load(args.ori_model_ckpt, map_location=device).state_dict()
        state_dict = torch.load(args.ori_model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    print('load model done')

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    if args.ori_opt_ckpt:
        print(args.ori_opt_ckpt)
        # optimizer_state_dict = torch.load(args.ori_opt_ckpt).state_dict()
        optimizer_state_dict = torch.load(args.ori_opt_ckpt)
        optimizer.load_state_dict(optimizer_state_dict)
    print('set optimizer done')

    # load val data
    val_dataset = None
    val_loader = None
    if args.evaluate_per_epoch > 0:
        val_dataset = Spk251_test(spk_ids, args.root, return_file_name=True, wav_length=None)
        test_loader_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
        }
        val_loader = DataLoader(val_dataset, **test_loader_params)
        
    # load backdoor val data
    backdoor_val_dataset = None
    backdoor_val_loader = None
    if args.evaluate_per_epoch > 0:
        backdoor_val_dataset = backdoor_Spk251_test(spk_ids, args.root, return_file_name=True, wav_length=None)
        test_loader_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
        }
        backdoor_val_loader = DataLoader(backdoor_val_dataset, **test_loader_params)

    # load train data
    train_dataset = Spk251_train(spk_ids, args.root, wav_length=args.wav_length)
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': False 
    }
    train_loader = DataLoader(train_dataset, **train_loader_params)
    print('load train data done', len(train_dataset))
    
    # load backdoor train data
    backdoor_train_dataset = backdoor_Spk251_train(spk_ids, args.root, wav_length=args.wav_length)
    backdoor_train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': False 
    }
    backdoor_train_loader = DataLoader(backdoor_train_dataset, **backdoor_train_loader_params)
    print('load train data done', len(backdoor_train_dataset))


    # loss
    criterion = torch.nn.CrossEntropyLoss()
    
    # 
    log = args.log if args.log else './model_file/AuioNet-adver-{}-{}.log'.format(args.defense, args.defense_param)
    logging.basicConfig(filename=log, level=logging.DEBUG)
    model_ckpt = args.model_ckpt if args.model_ckpt else './model_file/AudioNet-adver-{}-{}'.format(args.defense, args.defense_param)
    print(log, model_ckpt)

    num_batches = len(train_dataset) // args.batch_size
    
    
    ###################################################################################
    ###################################################################################
    ###################################################################################
    model1=model
    model_dict1 = model.state_dict()
    model_dict=[]
    optimizer_attack_temp = optimizer
    optimizer_benign_temp = optimizer
    adv_client = [1]

    i=-1
    for name1,data1 in model_dict1.items():
        i=i+1
        model_dict.append(data1.cpu().numpy())
    
    for i_epoch in range(args.num_epoches):
        global_model = []
        
        for i_client in range(0, 10):
            model = model1
            models = model.modules()
            for p in models:
                if p._get_name()!='Linear':
                    p.requires_grad_=False
            
            if i_client in adv_client:
                all_accuracies = []
                model.train()
                if optimizer_attack_temp:
                    optimizer = optimizer_attack_temp
                for batch_id, (x_batch, y_batch) in enumerate(backdoor_train_loader):
                    start_t = time.time()
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    # print(x_batch.min(), x_batch.max())

                    #Gaussian augmentation to normal samples
                    all_ids = range(x_batch.shape[0])
                    normal_ids = all_ids

                    if args.aug_eps > 0.:
                        x_batch_normal = x_batch[normal_ids, ...]
                        y_batch_normal = y_batch[normal_ids, ...]

                        a = np.random.rand()
                        noise = torch.rand_like(x_batch_normal, dtype=x_batch_normal.dtype, device=device)
                        epsilon = args.aug_eps
                        noise = 2 * a * epsilon * noise - a * epsilon
                        x_batch_normal_noisy = x_batch_normal + noise
                        x_batch = torch.cat((x_batch, x_batch_normal_noisy), dim=0)
                        y_batch = torch.cat((y_batch, y_batch_normal))
                    
                    outputs = model(x_batch)
                    
                    loss = criterion(outputs, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_attack_temp = optimizer
                    # print('main:', x_batch.min(), x_batch.max())

                    predictions, _ = model.make_decision(x_batch)
                    acc = torch.where(predictions == y_batch)[0].size()[0] / predictions.size()[0]

                    end_t = time.time() 
                    if batch_id % 10 == 0:
                        print("Batch", batch_id, "/", num_batches, ": Acc = ", round(acc,4), "\t batch time =", end_t-start_t, end='\r')
                    all_accuracies.append(acc)

                print()
                print('--------------------------------------') 
                print(f'Client : {i_client + 1} / 10')
                print("EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, ": Backdoor Acc = ", round(np.mean(all_accuracies),4), ': Loss = ', loss,  ': aggregate_lr = ', args.lr)
                print('--------------------------------------') 
                print()
                logging.info("EPOCH {}/{}: Acc = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(all_accuracies)))
                if args.wandb_start == 1:
                    wandb.log({'backdoor_client_acc':acc, 'backdoor_client_loss': loss, 'epoch':i_epoch})
                
                ### save ckpt
                ckpt = model_ckpt + "_{}".format(i_epoch + args.start_epoch)
                ckpt_optim = ckpt + '.opt'
                # torch.save(model, ckpt)
                # torch.save(optimizer, ckpt_optim)
                torch.save(model.state_dict(), ckpt)
                torch.save(optimizer.state_dict(), ckpt_optim)
                print()
                print("Save epoch ckpt in %s" % ckpt)
                print()

                if args.val == 1:
                    ### evaluate
                    if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
                        val_acc = validation_benign(model, val_loader) 
                        print()
                        print('benign Val Acc in attack client: %f' % (val_acc))
                        print()
                        logging.info('benign Val Acc in attack client: {:.6f}'.format(val_acc))
                    
                    ### evaluate
                    if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
                        val_acc = validation_backdoor(model, backdoor_val_loader) 
                        print()
                        print('Backdoor Val Acc in bening client: %f' % (val_acc))
                        print()
                        logging.info('Backdoor Val Acc in bening client: {:.6f}'.format(val_acc))
                
                # 提取每个客户机的权重加入global_model
                global_model.append(model.state_dict())

            if i_client not in adv_client:
                all_accuracies = []
                model.train()
                if optimizer_benign_temp:
                    optimizer = optimizer_benign_temp
                for batch_id, (x_batch, y_batch) in enumerate(train_loader):
                    start_t = time.time()
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    # print(x_batch.min(), x_batch.max())

                    #Gaussian augmentation to normal samples
                    all_ids = range(x_batch.shape[0])
                    normal_ids = all_ids

                    if args.aug_eps > 0.:
                        x_batch_normal = x_batch[normal_ids, ...]
                        y_batch_normal = y_batch[normal_ids, ...]

                        a = np.random.rand()
                        noise = torch.rand_like(x_batch_normal, dtype=x_batch_normal.dtype, device=device)
                        epsilon = args.aug_eps
                        noise = 2 * a * epsilon * noise - a * epsilon
                        x_batch_normal_noisy = x_batch_normal + noise
                        x_batch = torch.cat((x_batch, x_batch_normal_noisy), dim=0)
                        y_batch = torch.cat((y_batch, y_batch_normal))

                    outputs = model(x_batch)
                    
                    loss = criterion(outputs, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_benign_temp = optimizer
                    # print('main:', x_batch.min(), x_batch.max())

                    predictions, _ = model.make_decision(x_batch)
                    acc = torch.where(predictions == y_batch)[0].size()[0] / predictions.size()[0]

                    end_t = time.time() 
                    if batch_id % 10 == 0:
                        print("Batch", batch_id, "/", num_batches, ": Acc = ", round(acc,4), "\t batch time =", end_t-start_t, end='\r')
                    all_accuracies.append(acc)
                
                print()
                print('--------------------------------------') 
                print(f'Client : {i_client + 1} / 10')
                print("EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, ": Acc = ", round(np.mean(all_accuracies),4), ': Loss = ', loss,  ': aggregate_lr = ', args.lr)              
                print('--------------------------------------') 
                print()
                logging.info("EPOCH {}/{}: Acc = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(all_accuracies)))

                ### save ckpt
                ckpt = model_ckpt + "_{}".format(i_epoch + args.start_epoch)
                ckpt_optim = ckpt + '.opt'
                # torch.save(model, ckpt)
                # torch.save(optimizer, ckpt_optim)
                torch.save(model.state_dict(), ckpt)
                torch.save(optimizer.state_dict(), ckpt_optim)
                print()
                print("Save epoch ckpt in %s" % ckpt)
                print()

                if args.val == 1:
                    ### evaluate
                    if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
                        val_acc = validation_benign(model, val_loader) 
                        print()
                        print('benign Val Acc in bening client: %f' % (val_acc))
                        print()
                        logging.info('benign Val Acc in bening client: {:.6f}'.format(val_acc))
                    
                    ### evaluate
                    if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
                        val_acc = validation_backdoor(model, backdoor_val_loader) 
                        print()
                        print('Backdoor Val Acc in bening client: %f' % (val_acc))
                        print()
                        logging.info('Backdoor Val Acc in bening client: {:.6f}'.format(val_acc))
                
                for name,data in model.state_dict().items():
                    data = data.clone().detach().requires_grad_(False)
                
                # 提取每个客户机的权重加入global_model
                global_model.append(model.state_dict())
                
        # 以上，将每个客户机的模型保存在了global_model[]中
        # 以下，从global_model[]中读取模型的参数，进行聚合，更新至model
        #在这里写聚合，其中 global_model_state_dict[] 里面有10个客户机保存下来的模型
        print('开始聚合')
        for i_clients in range(0, 10):
            i=-1
            ori=global_model[i_clients]
            dic_data=[]
            for name, data in ori.items():
                i = i + 1
                dic_data.append(copy.deepcopy(data.cpu().numpy()))
            if i_clients in adv_client:
                delta = np.array(dic_data) - np.array(model_dict)
                delta = delta * 10
            else:
                delta = np.array(dic_data) - np.array(model_dict)
            #delta = dic_data - model_dict.numpy()

        delt_av=(delta) * (1/10)
        new_weights = np.array(model_dict) - np.array(delt_av) * args.lr
        # new_weights = new_weights.cpu().numpy()
        model2=model1.state_dict()

        print('更新全局模型')
        i=-1
        for name,data in model2.items():
            i=i+1
            # 让全局模型+当前更新（经过1/n和*lr的变化），成为新的全局模型
            data = np.float64(data.cpu())
            data += (new_weights[i])
        
        model.load_state_dict(model2,strict=False)
        
        ### evaluate
        if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
            val_acc = validation_benign(model, val_loader) 
            print()
            print('benign Val Acc in global model: %f' % (val_acc))
            print()
            logging.info('benign Val Acc in global model: {:.6f}'.format(val_acc))
            if args.wandb_start == 1:
                wandb.log({'benign_global_acc':val_acc, 'epoch':i_epoch})
        
        ### evaluate
        if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
            val_acc = validation_backdoor(model, backdoor_val_loader) 
            print()
            print('Backdoor Val Acc in global model: %f' % (val_acc))
            print()
            logging.info('Backdoor Val Acc in global model: {:.6f}'.format(val_acc))
            if args.wandb_start == 1:
                wandb.log({'backdoor_global_acc':val_acc, 'epoch':i_epoch})
            
        
    ###################################################################################
    ###################################################################################
    ###################################################################################
        
    # torch.save(model, model_ckpt)
    torch.save(model.state_dict(), model_ckpt)

    if args.wandb_start == True:
        wandb.finish()

if __name__ == '__main__':

    main(parser_args())
    endtime = time.time()
    dtime = endtime - starttime
    print("time：  %.8s s" % dtime)
    