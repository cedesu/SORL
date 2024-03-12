from time import perf_counter
from argparse import ArgumentParser
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.atari_dataloader import MultiprocessAtariDataLoader
from utils.atari_head_dataloader_traj import MultiprocessAtariHeadDataLoader

from utils.networks import Mnih2015_mh_lora as Mnih2015

import gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from PIL import Image, ImageChops
from functools import reduce
from random import randint
from torch.distributions.categorical import Categorical
import os

accum_n=5
n_head=3

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk


class PPOAgent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1=nn.Conv2d(3, 32, 8, stride=4)
        self.conv2=nn.Conv2d(32, 64, 4, stride=2)
        self.conv3=nn.Conv2d(64, 64, 3, stride=1)
        
        self.fc1=nn.Linear(64 * 7 * 7, 512)
        
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def get_value(self, x):
        #x=x/255#x/=255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 64*7*7)
        x=F.relu(self.fc1(x))
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        #x=x/255#x/=255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 64*7*7)
        x=F.relu(self.fc1(x))
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    parser = ArgumentParser("Train PyTorch models to do imitation learning.")
    parser.add_argument("input_directory", type=str,
                        help="Path to directory with recorded gameplay.")
    parser.add_argument("game", type=str,
                        help="Name of the game to use for training.")
    parser.add_argument("model", nargs="?", type=str,
                        help="Path of the file where model will be saved.") 
    parser.add_argument("--actions", type=int, default=18,
                        help="Number of actions")       
    parser.add_argument("--framestack", type=int, default=3,
                        help="Number of frames to stack")
    parser.add_argument("--merge", action="store_true",
                        help="Merge stacked frames into one image.")
    parser.add_argument("--width", "-x", type=int, default=84,
                        help="Width of the image")
    parser.add_argument("--height", "-y", type=int, default=84,
                        help="Height of the image")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes to use for the dataloader.")
    parser.add_argument("--l2", type=float, default="0.00001",
                        help="L2 regularization weight.")
    parser.add_argument("--percentile", type=int,
                        help="The top q-percentile of samples to use for training.")
    parser.add_argument("--top-n", type=int,
                        help="The top n number of samples to use for training.")
    parser.add_argument("--save-freq", type=int, default=1,
                        help="Number of epochs between checkpoints.")
    parser.add_argument("--augment", action="store_true",
                        help="Use image augmentation.")
    parser.add_argument("--preload", action="store_true",
                        help="Preload image data to memory.")
    parser.add_argument("--atari-head", action="store_true",
                        help="Use the Atari-HEAD dataloader.")
    parser.add_argument("--action-delay", type=int, default=0,
                        help="How many frames to delay the actions by.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Don't use CUDA")
    parser.add_argument("--json", action="store_true",
                        help="Dataset is stored as JSON")
    parser.add_argument("--fileformat", type=str, default="png",
                        help="Postfix of the image files to be loaded")
    parser.add_argument("--no-op", type=int, default=0,
                    help="Maximum number of no-op actions at the beginning of each game.")
    parser.add_argument("--action", type=str, default="sampling",
                    choices=["sampling", "argmax"],
                    help="Use random sampling or argmax to pick actions.")
    parser.add_argument("--max-frames", type=int, default=40000,
                    help="Maximum number of frames to run the game for before ending evaluation.")  
    parser.add_argument("--load_ppo", type=str, default='/home/wjh/myh/video_bc/models_atari_head/dqn_models/space_dqn/_30.pt',
                    help="Maximum number of frames to run the game for before ending evaluation.")   
    parser.add_argument("--env_id", type=str, default=None,
                    help="Maximum number of frames to run the game for before ending evaluation.")   

    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name=args.env_id#"MsPacman-v0"#"SpaceInvaders-v0"#"BreakoutNoFrameskip-v4"#"Breakout-v0"#
    env=gym.make(env_name, full_action_space=True)
    rng = np.random.default_rng()
    
    # ppo
    # ppo_network = PPOAgent(args.actions).to(device)
    # assert args.load_ppo!=''
    # ppo_network.load_state_dict(torch.load(args.load_ppo))
    ppo_network = torch.load(args.load_ppo)

    model_cls = Mnih2015(
        (args.width, args.height),
        3 if args.merge else 3*args.framestack,
        args.actions,
        n_head=n_head
    ).to(device)
    optimizer_cls = torch.optim.Adam(model_cls.parameters(), weight_decay=args.l2)
    
    # model_cls.conv1.weight=ppo_network.conv1.weight
    # model_cls.conv2.weight=ppo_network.conv2.weight
    # model_cls.conv3.weight=ppo_network.conv3.weight
    # model_cls.conv1.bias=ppo_network.conv1.bias
    # model_cls.conv2.bias=ppo_network.conv2.bias
    # model_cls.conv3.bias=ppo_network.conv3.bias
    # model_cls.fc1.weight=ppo_network.fc1.weight
    # model_cls.fc2.weight=ppo_network.actor.weight
    # model_cls.fc1.bias=ppo_network.fc1.bias
    # model_cls.fc2.bias=ppo_network.actor.bias
    
    model_adv = Mnih2015(
        (args.width, args.height),
        3 if args.merge else 3*args.framestack,
        args.actions,
        n_head=n_head
    ).to(device)
    optimizer_adv = torch.optim.Adam(model_adv.parameters(), weight_decay=args.l2)
    
    # model_adv.conv1.weight=ppo_network.conv1.weight
    # model_adv.conv2.weight=ppo_network.conv2.weight
    # model_adv.conv3.weight=ppo_network.conv3.weight
    # model_adv.conv1.bias=ppo_network.conv1.bias
    # model_adv.conv2.bias=ppo_network.conv2.bias
    # model_adv.conv3.bias=ppo_network.conv3.bias
    # model_adv.fc1.weight=ppo_network.fc1.weight
    # model_adv.fc2.weight=ppo_network.actor.weight
    # model_adv.fc1.bias=ppo_network.fc1.bias
    # model_adv.fc2.bias=ppo_network.actor.bias
    
    model_eval=model_cls

    dataloader_args = {
        "directory": args.input_directory,
        "game": args.game,
        "stack": args.framestack,
        "batch_size": args.batch,
        "size": (args.width, args.height),
        "percentile": args.percentile,
        "top_n": args.top_n,
        "augment": args.augment,
        "preload": args.preload,
        "merge": args.merge,
        "json": args.json,
        "action_delay": args.action_delay,
        "fileformat": args.fileformat
    }

    # Note: if new dataloader arguments are added, make sure they work with
    #       both loaders, or if they don't, remove them with 'del' below
    if args.atari_head:
        del dataloader_args["game"]
        del dataloader_args["json"]
        del dataloader_args["fileformat"]
        gen = MultiprocessAtariHeadDataLoader(dataloader_args, args.workers)
    else:
        gen = MultiprocessAtariDataLoader(dataloader_args, args.workers)
    shape = gen.shape

    history = dict()
    history["loss"] = []
    history["accuracy"] = []
    
    #os.mkdir(args.model)
    
    for epoch in range(1, args.epochs + 1):
        if epoch<args.epochs//2+1:
            model_eval=model_cls
        else:
            model_eval=model_adv
        
        
        # change the worst model
        # if epoch>1:
        #     '''analyze_z=np.array(analyze_z).mean(0)
        #     best=int(analyze_z.argmax())
        #     worst=int(analyze_z.argmin())
        #     print('change the worst',worst,'to the best',best)
        #     model_cls.lora1_up[worst].load_state_dict(model_cls.lora1_up[best].state_dict())
        #     model_cls.lora1_down[worst].load_state_dict(model_cls.lora1_down[best].state_dict())
        #     model_cls.lora2_up[worst].load_state_dict(model_cls.lora2_up[best].state_dict())
        #     model_cls.lora2_down[worst].load_state_dict(model_cls.lora2_down[best].state_dict())'''
            
        #     analyze_z=np.array(analyze_z).argmax(1)
        #     analyze_z=np.array([(analyze_z==i).mean() for i in range(n_head)])
        #     best=int(analyze_z.argmax())
        #     worst=int(analyze_z.argmin())
        #     if analyze_z[worst]<1/n_head/2:
        #         print('change the worst',worst,'to the best',best,analyze_z)
        #         model_cls.lora1_up[worst].load_state_dict(model_cls.lora1_up[best].state_dict())
        #         model_cls.lora1_down[worst].load_state_dict(model_cls.lora1_down[best].state_dict())
        #         model_cls.lora2_up[worst].load_state_dict(model_cls.lora2_up[best].state_dict())
        #         model_cls.lora2_down[worst].load_state_dict(model_cls.lora2_down[best].state_dict())
        #     else:
        #         print('unchanged',analyze_z)
                
                
        print("Starting epoch {}".format(epoch))
        model_cls.train()
        model_adv.train()
        start = perf_counter()

        # Accuracy
        correct = 0
        total = 0

        # Loss
        loss_sum = 0
        loss_num = 0

        accum_cnt=0
        accum_loss_cls=0
        accum_loss_adv=0
        gen.switch_mode(True)
        analyze_z=[]
        avg_z=torch.zeros(3)#.cuda()
        for batch, data in enumerate(gen):
            # Convert data to correct format
            x = torch.Tensor(np.swapaxes(data[0], 1, 3)).to(device) / 255
            if args.json:
                raise
                # Drop unnecessary axis
                y = torch.Tensor(data[1]).to(device)[:, 0, :]
            else:
                y = torch.argmax(torch.Tensor(data[1]).to(device), 1).long()

            # Get model output
            z=model_cls.select(x,y)#[1,0,0]#
            
            #rebalance
            # avg_z=avg_z*0.98+0.02*z.clone()
            # avg_z=avg_z/avg_z.sum()
            # z=z/avg_z
            # z=z/z.sum()
            #z=torch.tensor([1.,0.,0.])
            
            analyze_z.append(z.cpu().numpy())
            
            if epoch<args.epochs//2+1:
                # output = model_cls(x,z)
                
                #way2
                output = model_cls.forward_all_head(x,z)
                output=(F.softmax(output,dim=-1).clamp(0.001,1000)*z.cuda().unsqueeze(1).unsqueeze(1)).sum(dim=0)#F.softmax(output,dim=-1)[0]#
                
                # Calculate accuracy and add to epoch statistics
                if args.json:
                    correct += 0 # TODO
                else:
                    correct += output.argmax(1).eq(y).sum()

                total += len(y)
                
                # Calculate loss cls
                if args.json:
                    raise
                    loss = F.binary_cross_entropy_with_logits(output, y)
                else:
                    #loss = F.cross_entropy(output, y)
                    
                    # loss_cls=(-F.softmax(output,-1).log().gather(1,y.unsqueeze(1))).mean()
                    
                    #way2
                    loss_cls=(-output.log().gather(1,y.unsqueeze(1))).mean()
                    
                # Add loss to epoch statistics
                loss_sum += loss_cls
                loss_num += 1
                
                if accum_cnt==accum_n:
                    # Backpropagate loss
                    optimizer_cls.zero_grad()
                    accum_loss_cls.backward()
                    optimizer_cls.step()
                    
                    accum_cnt=0
                    accum_loss_cls=0
                else:
                    accum_cnt+=1
                    accum_loss_cls+=loss_cls
            else:
                # Calculate loss adv
                loss_adv=0
                adv=ppo_network(x).gather(-1,y.unsqueeze(1))[:,0]#_,_,_,adv=ppo_network.get_action_and_value(x,y)
                adv=adv-adv.mean(-1)
                lmd=1
                adv=adv/lmd
                adv=adv.softmax(dim=0)
                for i in range(n_head):
                    z_i=torch.zeros(n_head)#torch.zeros_like(z)
                    z_i[i]=1
                    pred_i=model_adv(x,z_i)
                    #print('db',pred_i.shape,y.shape,x.shape)
                    loss_adv+=(-F.softmax(pred_i,-1).clamp(1e-4,1).log().gather(1,y.unsqueeze(1)).squeeze(1)*adv*z[i]).sum()
                    #loss_adv+=(-F.softmax(pred_i,-1).clamp(1e-4,1).log().gather(1,y.unsqueeze(1)).squeeze(1)*z[i]).sum()
                    
                # #gaidaoyiban
                # pred = model_adv.forward_all_head(x)
                # prob=(F.softmax(pred,-1)*z.unsqueeze(1).unsqueeze(2)).sum(dim=0)
                # actor_loss_cls = (-prob.log().gather(1,data.actions)).mean()

                # Add loss to epoch statistics
                loss_sum += loss_adv
                loss_num += 1
                
                if accum_cnt==accum_n:
                    # Backpropagate loss
                    optimizer_adv.zero_grad()
                    accum_loss_adv.backward()
                    optimizer_adv.step()
                    
                    accum_cnt=0
                    accum_loss_adv=0
                else:
                    accum_cnt+=1
                    accum_loss_adv+=loss_adv
                    
                # Calculate accuracy and add to epoch statistics
                with torch.no_grad():
                    output=model_adv(x,z)
                if args.json:
                    correct += 0 # TODO
                else:
                    correct += output.argmax(1).eq(y).sum()

                total += len(y)

            # Print statistics
            if batch % 2000 == 0:
                end = perf_counter()
                accuracy = float(correct) / float(total)
                loss = loss_sum / loss_num
                print("Epoch {} - {}/{}: loss: {}, train acc: {} ({} s/batch)".format(
                    epoch,
                    batch,
                    len(gen),
                    loss,
                    accuracy,
                    (end - start) / 100)
                )
                start = perf_counter()
        gen.switch_mode(False)
        zs,kl_list=[],[]
        correct,total=0,0
        for batch, data in enumerate(gen):
            # Convert data to correct format
            x = torch.Tensor(np.swapaxes(data[0], 1, 3)).to(device) / 255
            if args.json:
                raise
                # Drop unnecessary axis
                y = torch.Tensor(data[1]).to(device)[:, 0, :]
            else:
                y = torch.argmax(torch.Tensor(data[1]).to(device), 1).long()

            # Get model output
            z=model_cls.select(x,y)#torch.tensor([1,0,0])#
            z_show=model_eval.select(x,y)
            zs.append(z_show.cpu().numpy())
            
            # output = model_eval(x,z)
                
            #way2
            output = model_eval.forward_all_head(x,z)
            
            # calculate kl
            kl=0
            for i in range(n_head):
                for j in range(n_head):
                    if i!=j:
                        kl+=F.kl_div(F.log_softmax(output[i],-1),F.softmax(output[j],-1))
            kl_list.append(kl.detach().cpu())
            
            output=(F.softmax(output,dim=-1).clamp(0.001,1000)*z.cuda().unsqueeze(1).unsqueeze(1)).sum(dim=0)#F.softmax(output,dim=-1)[0]#

            # Calculate accuracy and add to epoch statistics
            if args.json:
                correct += 0 # TODO
            else:
                correct += output.argmax(1).eq(y).sum()

            total += len(y)
        print('show z',zs[:10],'kl',np.mean(kl_list))
        if epoch==30 or epoch==60:
            np.save('zs_'+args.env_id+'_'+str(epoch)+'_4',zs)

        # Print statistics
        end = perf_counter()
        accuracy = float(correct) / float(total)
        print("Epoch {} - {}: eval acc: {} ({} s/batch)".format(
            epoch,
            len(gen),
            accuracy,
            (end - start) / 100)
        )
        start = perf_counter()

        # Save statistics
        accuracy = float(correct) / float(total)
        loss = loss_sum / loss_num

        history["accuracy"].append(float(accuracy))
        history["loss"].append(float(loss))

        # with open(args.model + "-history.json", "w") as f:
        #     json.dump(history, f)

        # Save model
        # if False and epoch==1 or epoch%10==0:
        #     filename = "{}/{}".format(args.model, epoch)
        #     print("Saving {}".format(filename))
        #     torch.save(model_cls.state_dict(), filename+'_cls')
        #     torch.save(model_adv.state_dict(), filename+'_adv')

        #eval
        r_mean_list=[]
        for head in range(n_head):
            n_eval=10
            r_list=[]
            z_onehot=[0]*n_head
            z_onehot[head]=1
            for _ in range(n_eval):
                no_ops = randint(0, args.no_op)
                no_ops_done = 0
                
                o = env.reset()
                r, d, i = (0.0, False, None)

                total_reward = 0
                total_frames = 0

                # Create a frame stack and fill it with zeros (black images)
                stack = []
                for _ in range(args.framestack):
                    stack.append(np.zeros((args.width, args.height, 3), dtype=np.uint8))

                while True:
                    # Resize image
                    #print('resize',o.shape,env.observation_space,env.action_space)
                    img = Image.fromarray(o)
                    img = img.resize((args.width, args.height), Image.BILINEAR)
                    img = np.asarray(img)

                    # Update the frame stack
                    stack.insert(0, img)
                    while len(stack) > args.framestack:
                        stack.pop()

                    # Make sure we have enough frames stacked
                    if len(stack) != args.framestack:
                        continue

                    
                    if args.merge:
                        # Convert numpy arrays to images
                        image_stack = map(Image.fromarray, stack)

                        # Get lightest pixel values from the stack
                        img = reduce(ImageChops.lighter, image_stack)

                        np_stack = np.asarray(img, dtype=np.float32)
                        np_stack = np.expand_dims(np_stack, axis=0)
                    else:
                        # Convert stack to numpy array with correct dimensions and type
                        np_stack = np.concatenate(stack, axis=2)
                        np_stack = np.expand_dims(np_stack, axis=0)
                        np_stack = np_stack.astype(np.float32)

                    # Normalize
                    np_stack /= 255

                    if no_ops_done < no_ops:
                        # Send a no-op action if we haven't done enough no-ops yet
                        o, r, d, i = env.step(0)
                        no_ops_done += 1
                    
                    else:# not args.random:
                        prediction = model_eval(torch.Tensor(np.swapaxes(np_stack, 1, 3)).to(device),z_onehot).detach().cpu()
                        prediction = F.softmax(prediction, dim=1)

                        if args.action == "argmax":
                            prediction = np.argmax(prediction)
                        elif args.action == "sampling":
                            # Perform a weighted selection from the indices
                            prediction = np.array(prediction[0])
                            p = prediction/np.sum(prediction)
                            prediction = rng.choice(list(range(len(prediction))), p=p)

                        o, r, d, i = env.step(prediction)
                    #elif args.random:
                    #    o, r, d, i = env.step(np.random.randint(18))

                    total_reward += r
                    total_frames += 1

                    # Stop evaluation if game reaches terminal state or
                    # maximum number of frames is exceeded
                    if d or total_frames > args.max_frames:
                        r_list.append(total_reward)#print('total_reward',total_reward)
                        break
            r_mean_list.append(np.mean(r_list))
        print('total_reward of heads',r_mean_list)
        print("Finished epoch {}".format(epoch))
    
    gen.stop()
