## Let's go hard: sequences 

Sequences are a hard thing, because now what you need to do is to predict future events bases on the old ones,
and this kind of data can not be processed very well by simple linear layers

## Music: MIDI

This format can encode songs by sequential notes, each note is composed by:

- Pitch: a value between 0 and 127. You can imagine that as a piano with 128 keys
- Start: second where the note should start
- End: second where the note should end
- Velocity: how loud the note is played (can take values from 0 and 127)

## The machine learning problem.

So, your job is to train a model that learns a specific author distribution that can predict the next note
based on a seed

## How to train a model like this?

### The data

As you can notice, we can interpret this as temporal sequence, so inspired by natural language process methods 
we can re-interpret a song as a patch of n-grams:

p(notes[i] | notes[i-1]...notes[i-n])

### The model

In deep learning there are a special kind of layers that are specialized on handle sequential data: recurrent neural networks

### LSTM

Make no bones about, LSTM is one of the best implementations under the concept of recurrent neural networks, so in advance
i say, this is what we are going to use 