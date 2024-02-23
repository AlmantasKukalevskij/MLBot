"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""
import os
import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)

def get_last_episode(model_dir='models', model_name='model_debug2'):
    print(f"Looking in directory: {model_dir} for models with name: {model_name}")
    model_files = os.listdir(model_dir)
    print(f"Found model files: {model_files}")
    episodes = [
        int(filename.split('_')[-1].replace('.h5', ''))
        for filename in model_files
        if filename.startswith(model_name) and filename.endswith('.h5') and filename.split('_')[-1].replace('.h5', '').isdigit()
    ]
    print(f"Extracted episodes: {episodes}")
    last_episode = max(episodes) if episodes else 0
    print(f"Last episode: {last_episode}")
    return last_episode

def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="double-dqn", model_name="model_debug2", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    # Determine last episode number from saved models
    last_episode = get_last_episode(model_name=model_name)
    if last_episode > 0:
        print(f"Resuming from episode {last_episode + 1}")
        agent = Agent(window_size, strategy=strategy, pretrained=True, model_name=model_name, episode=last_episode)
    else:
        print("Starting training from scratch")
        agent = Agent(window_size, strategy=strategy, pretrained=False, model_name=model_name)



    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    # Adjust the range to start from the next episode after the last saved one
    for episode in range(last_episode + 1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)
    #save the model every 1 episodes
        if episode % 1 == 0:
            agent.save(episode)

    #save the final model after training is completed
    agent.save(ep_count)

if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
