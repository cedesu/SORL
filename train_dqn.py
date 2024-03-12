from time import perf_counter
from argparse import ArgumentParser
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.atari_dataloader import MultiprocessAtariDataLoader
from utils.atari_head_dataloader import MultiprocessAtariHeadDataLoader

from utils.networks import Mnih2015


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
    parser.add_argument("--env_id", type=str, default=None,
                        help="Postfix of the image files to be loaded")

    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mnih2015(
        (args.width, args.height),
        3 if args.merge else 3*args.framestack,
        args.actions
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2)
    
    model_target = Mnih2015(
        (args.width, args.height),
        3 if args.merge else 3*args.framestack,
        args.actions
    ).to(device)
    model_target.load_state_dict(model.state_dict())

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
        "fileformat": args.fileformat,
        'dqn':True
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
    
    env_name=args.env_id#"MontezumaRevenge-v0"#"SpaceInvaders-v0"#"Breakout-v0"#"MsPacman-v0"#
    env=gym.make(env_name, full_action_space=True)
    rng = np.random.default_rng()

    for epoch in range(1, args.epochs + 1):
        #eval
        n_eval=10
        r_list=[]
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
                    prediction = model(torch.Tensor(np.swapaxes(np_stack, 1, 3)).to(device)).detach().cpu()
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
        print('total_reward',np.mean(r_list),np.std(r_list))
        
        print("Starting epoch {}".format(epoch))
        model.train()
        start = perf_counter()

        # Accuracy
        correct = 0
        total = 0

        # Loss
        loss_sum = 0
        loss_num = 0

        for batch, data in enumerate(gen):
            # Convert data to correct format
            x = torch.Tensor(np.swapaxes(data[0], 1, 3)).to(device) / 255
            if args.json:
                # Drop unnecessary axis
                y = torch.Tensor(data[1]).to(device)[:, 0, :]
            else:
                y = torch.argmax(torch.Tensor(data[1]).to(device), 1).long()

            optimizer.zero_grad()

            # Get model output
            output = model(x)

            # Calculate loss
            if args.json:
                loss = F.binary_cross_entropy_with_logits(output, y)
            else:
                #loss = F.cross_entropy(output, y)
                #loss=(-F.softmax(output,-1).log().gather(1,y.unsqueeze(1))).mean()
                
                reward=torch.tensor(data[3]).to(device)
                next_obs=torch.Tensor(np.swapaxes(data[2], 1, 3)).to(device) / 255
                done=torch.tensor(data[4]).to(device)
                target_q=torch.max(model_target(next_obs),dim=-1).values
                gamma=0.99
                q=reward+done*gamma*output.gather(-1,y.unsqueeze(1))[:,0]
                loss=F.mse_loss(q,target_q.detach())
                
                #loss+=(output.exp().sum(-1).log()-output.gather(-1,y.unsqueeze(1))[:,0]).mean()

            # Add loss to epoch statistics
            loss_sum += loss
            loss_num += 1

            # Calculate accuracy and add to epoch statistics
            if args.json:
                correct += 0 # TODO
            else:
                correct += output.argmax(1).eq(y).sum()

            total += len(y)
            
            # Backpropagate loss
            loss.backward()
            optimizer.step()

            # Print statistics
            if batch % 3000 == 0:
                end = perf_counter()
                accuracy = float(correct) / float(total)
                loss = loss_sum / loss_num
                print("Epoch {} - {}/{}: loss: {}, acc: {} ({} s/batch)".format(
                    epoch,
                    batch,
                    len(gen),
                    loss,
                    accuracy,
                    (end - start) / 100)
                )
                start = perf_counter()

        # Save statistics
        accuracy = float(correct) / float(total)
        loss = loss_sum / loss_num

        history["accuracy"].append(float(accuracy))
        history["loss"].append(float(loss))

        with open(args.model + "-history.json", "w") as f:
            json.dump(history, f)

        # Save model
        if args.model is not None and epoch % args.save_freq == 0:
            filename = "{}_{}.pt".format(args.model, epoch)
            print("Saving {}".format(filename))
            torch.save(model, filename)
        
        model_target.load_state_dict(model.state_dict())

        print("Finished epoch {}".format(epoch))
    
    gen.stop()
