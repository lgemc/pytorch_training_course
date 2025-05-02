import torch
import torch.nn.functional as F
import numpy as np
import pretty_midi
from data.sequences import PITCH_COLUMN, VELOCITY_COLUMN, DURATION_COLUMN, STEP_COLUMN


def predict(model, notes_amount, device, seed):
    model.to(device)
    model.eval()

    # Prepare seed sequence
    seed_tensor = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).to(device)

    # Initialize the sequence with the seed
    generated_sequence = seed.clone()

    # Generate the notes
    with torch.no_grad():
        input_sequence = seed_tensor

        for _ in range(notes_amount):
            # Get model output
            outputs = model(input_sequence, teacher_forcing_ratio=0.0)

            # Extract individual predictions
            pitch_pred = outputs['pitch'][:, -1, :]
            velocity_pred = outputs['velocity'][:, -1, :]
            step_pred = outputs['step'][:, -1, :]
            duration_pred = outputs['duration'][:, -1, :]
            # Convert predictions to values
            pitch = torch.argmax(pitch_pred, dim=-1).item()
            velocity = torch.argmax(velocity_pred, dim=-1).item()
            step = step_pred.squeeze(-1).item()
            duration = duration_pred.squeeze(-1).item()

            # Ensure step is not negative
            step = max(0, step)

            # Ensure duration is positive
            duration = max(0.01, duration)

            # Create the
            new_note = np.zeros_like(seed[0])
            new_note[PITCH_COLUMN] = pitch
            new_note[VELOCITY_COLUMN] = velocity
            new_note[STEP_COLUMN] = step
            new_note[DURATION_COLUMN] = duration


            # Add the new note to our sequence
            generated_sequence = np.vstack([generated_sequence, [new_note]])

            # Update input for next iteration - create a new seed with the latest generated note
            last_note = torch.tensor([new_note], dtype=torch.float32).unsqueeze(0).to(device)

            input_sequence = torch.cat((input_sequence[:, 1:, :], last_note), dim=1)
            input_sequence = input_sequence.to(device)

    return generated_sequence

def items_to_notes(items, step_max=None, step_min=None, duration_max=None, duration_min=None):
    """
    Convert a tensor from the dataset to a pretty_midi.Note objects
    """
    pitches = items[:, PITCH_COLUMN]
    velocities = items[:, VELOCITY_COLUMN]
    durations = items[:, DURATION_COLUMN]
    durations = durations * (duration_max - duration_min) + duration_min
    steps = items[:, STEP_COLUMN]
    steps = steps * (step_max - step_min) + step_min

    notes = []
    for i in range(len(pitches)):
        pitch = pitches[i].item()
        velocity = velocities[i].item()
        duration = durations[i].item()
        step = steps[i].item()

        start = 0 if i == 0 else notes[-1].end + step
        end = start + duration

        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(start),
            end=float(end),
        )

        notes.append(note)

    return notes