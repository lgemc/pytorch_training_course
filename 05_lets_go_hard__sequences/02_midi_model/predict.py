from typing import List

import torch
import pretty_midi

def note_to_tensor(
        note: pretty_midi.Note,
        step_max: float,
        step_min: float,
        duration_max: float,
        duration_min: float,
        prev_note: pretty_midi.Note = None,
) -> dict:
    """
    Convert a pretty_midi.Note object to a tensor.
    """
    step = note.start - (prev_note.start if prev_note else 0)
    duration = note.end - note.start
    step = (step - step_min) / (step_max - step_min)
    duration = (duration - duration_min) / (duration_max - duration_min)

    return {
        "step": torch.tensor([step], dtype=torch.float32),
        "duration": torch.tensor([duration], dtype=torch.float32),
        "pitch": torch.tensor([note.pitch], dtype=torch.long),
        "velocity": torch.tensor([note.velocity], dtype=torch.long),
    }

def notes_to_tensor(
        notes,
        step_max: float,
        step_min: float,
        duration_max: float,
        duration_min: float,
        device = "cpu",
) -> dict:
    """
    Convert a list of pretty_midi.Note objects to a tensor.
    """
    steps = []
    durations = []
    pitches = []
    velocities = []

    for note in notes:
        step = note.start - (notes[0].start if len(notes) > 1 else 0)
        duration = note.end - note.start
        step = (step - step_min) / (step_max - step_min)
        duration = (duration - duration_min) / (duration_max - duration_min)

        steps.append(step)
        durations.append(duration)
        pitches.append(note.pitch)
        velocities.append(note.velocity)

    return {
        "step": torch.tensor(steps, dtype=torch.float32).to(device),
        "duration": torch.tensor(durations, dtype=torch.float32).to(device),
        "pitch": torch.tensor(pitches, dtype=torch.long).to(device),
        "velocity": torch.tensor(velocities, dtype=torch.long).to(device),
    }

def tensor_to_note(
        out: dict,
        step_max: float,
        step_min: float,
        duration_max: float,
        duration_min: float,
        prev_note: pretty_midi.Note = None,
) -> pretty_midi.Note:
    """
    Convert a tensor to a pretty_midi.Note object.
    """
    pitch = torch.argmax(out["pitch"], dim=-1)
    velocity = torch.argmax(out["velocity"], dim=-1)
    duration = out["duration"].squeeze(-1).item()
    duration = duration * (duration_max - duration_min) + duration_min
    step = out["step"].squeeze(-1).item()
    step = step * (step_max - step_min) + step_min

    start = (prev_note.end if prev_note else 0) + step
    end = start + duration

    return pretty_midi.Note(
        start=float(start), end=float(end), pitch=int(pitch), velocity=int(velocity)
    )

def predict(
        model : torch.nn.Module,
        seed: List[pretty_midi.Note],
        notes_amount : int,
        device : str,
) -> List[pretty_midi.Note]:
    model.to(device)

    model.eval()

    notes = seed.copy()

    seed = notes_to_tensor(
        seed,
        step_max=model.step_max,
        step_min=model.step_min,
        duration_max=model.duration_max,
        duration_min=model.duration_min
    )


    with torch.no_grad():
        for i in range(notes_amount):
            seed = {k: v.to(device) for k, v in seed.items()}

            print(seed)

            out = model(seed)

            pitch = torch.argmax(out["pitch"], dim=-1)
            velocity = torch.argmax(out["velocity"], dim=-1)
            # Update the seed for the next iteration, moving the window, step and other are torch tensor, also output
            seed = {
                "step": torch.cat((seed["step"][1:], out["step"])),
                "duration":  torch.cat((seed["duration"][1:], out["duration"])),
                "pitch": torch.cat((seed["pitch"][1:], pitch.unsqueeze(-1))),
                "velocity": torch.cat((seed["velocity"][1:],velocity.unsqueeze(-1))),
            }

            note = tensor_to_note(
                    out,
                    step_max=model.step_max,
                    step_min=model.step_min,
                    duration_max=model.duration_max,
                    duration_min=model.duration_min,
                    prev_note=notes[-1]
            )

            notes.append(note)

    return notes

