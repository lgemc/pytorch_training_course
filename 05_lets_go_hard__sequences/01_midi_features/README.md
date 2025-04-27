# MIDI: features

As you can notice, we can't use start and end as it is because for the model this has no sense, 
but we can extract from there a feature that can be used in a model: step

`note.step = note.start - notes[i - 1].start`