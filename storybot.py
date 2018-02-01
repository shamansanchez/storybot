from __future__ import print_function

import discord
from discord.ext import commands
import asyncio
import os
import subprocess

import numpy as np
import tensorflow as tf

import time
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

bot = commands.Bot(command_prefix='s!')

storypath = "/home/jason/storybot/data"

def getStoryTypes():
    return [t for t in os.listdir(storypath)]

def sample(path, num=1000, start=u" "):
    tf.reset_default_graph()
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(path, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.sample(sess, chars, vocab, num, start, 1)

@bot.event
async def on_ready():
    print('Logged in as:')
    print(bot.user.name)
    print(bot.user.id)

@bot.command(description="testing")
async def echo(message : str):
    await bot.say(message)

@bot.command(description="Story Types")
async def types():
    mess = "I know the following stories: {0}".format(", ".join(getStoryTypes()))

    await bot.say(mess)

@bot.command(description="Tell a story")
async def story(t: str):
    if t not in getStoryTypes():
        await bot.say("I don't know that one...")
        return

    await bot.type()
    path = "{0}/{1}/out".format(storypath,t)
    story = sample(path)
    await bot.say('\n'.join(story.split('\n')[1:-1]))


bot.run("DISCORD TOKEN HERE")
