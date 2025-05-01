# Lets build a MIDI dataset based on ngrams 


# Important lessons from MIDI datasets

## How pass more than one input to the model

If we want to pass more than one input to the model, we can use a dictionary to pass the inputs. For example, if we want to pass the input and the target, we can do it like this:

```python
x = {
    'step': steps,
    'velocity': velocities,
    'duration': durations,
    'pitch': pitches,
}

y = {
    'step': steps,
    'velocity': velocities,
    'duration': durations,
    'pitch': pitches,
}
```

## Handling quantitative data and qualitative data

For example, in MIDI velocity and pitch are qualitative data (from 0 to 127), while step and duration are quantitative

With this in mind, both pitch and velocity should be passed by an embed layer before apply other kind of operation:

```python
torch.nn.Embedding(128, velocity_vocab_size)
```

And in order to get velocity and step in the same page with embed data and between them we should normalize them

### About normalization

#### where we should normalize data?

Of course, at dataset. It means that max and min values should be calculated inside this dataset, and at
`__get_item__` function we should normalize all data that should be normalized.

#### how to denormalize data (while predicting information)?

Nice, we should keep a copy of max and min data inside the model, in order to reuse them at perdict function

**Important note**: we should not normalize data inside the model, the data should arrive normalized to forward function.

So, predict function should have capabilities for convert one piece of data into tensors and for convert tensors
in data.


