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

def tensor_to_note(
        tensor: dict,
        step_max: float,
        step_min: float,
        duration_max: float,
        duration_min: float,
        prev_note: pretty_midi.Note = None,
) -> pretty_midi.Note:
    """
    Convert a tensor to a pretty_midi.Note object.
    """
    step = tensor["step"].item() * (step_max - step_min) + step_min
    duration = tensor["duration"].item() * (duration_max - duration_min) + duration_min
    pitch = tensor["pitch"].item()
    velocity = tensor["velocity"].item()

    start = (prev_note.end if prev_note else 0) + step
    end = start + duration

    return pretty_midi.Note(
        start=float(start), end=float(end), pitch=int(pitch), velocity=int(velocity)
    )

def predict(
        model : torch.nn.Module,
        seed: pretty_midi.Note,
        notes_amount : int,
        device : str,
) -> List[pretty_midi.Note]:
    model.to(device)

    model.eval()

    notes = [seed]

    seed = note_to_tensor(
        seed,
        step_max=model.step_max,
        step_min=model.step_min,
        duration_max=model.duration_max,
        duration_min=model.duration_min
    )

    with torch.no_grad():
        for i in range(notes_amount):
            seed = {k: v.to(device) for k, v in seed.items()}
            out = model(seed)
            pitch = torch.argmax(out["pitch"], dim=-1)
            velocity = torch.argmax(out["velocity"], dim=-1)
            duration = out["duration"].squeeze(-1).item()

            print(f"pitch: {pitch.item()}, velocity: {velocity.item()}, duration: {duration}")

            # it should start from the last note
            notes.append(
                tensor_to_note(
                    {
                        "step": seed["step"],
                        "duration": torch.tensor([duration], dtype=torch.float32),
                        "pitch": pitch,
                        "velocity": velocity,
                    },
                    step_max=model.step_max,
                    step_min=model.step_min,
                    duration_max=model.duration_max,
                    duration_min=model.duration_min,
                    prev_note=notes[-1]
                )
            )

            # update the seed for the next iteration
            seed = note_to_tensor(
                notes[-1],
                step_max=model.step_max,
                step_min=model.step_min,
                duration_max=model.duration_max,
                duration_min=model.duration_min,
                prev_note=notes[-2] if len(notes) > 1 else None
            )

    return notes

