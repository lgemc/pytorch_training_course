from typing import List

import torch
import pretty_midi

def predict(
        model : torch.nn.Module,
        notes_amount : int,
        device : str,
        seed : dict
) -> List[pretty_midi.Note]:
    model.to(device)

    model.eval()
    notes = [
        pretty_midi.Note(
            start=seed["step"][0].item(),
            end=seed["duration"][0].item(),
            pitch=seed["pitch"][-1].item(),
            velocity=seed["velocity"][-1].item()
        )
    ]

    with torch.no_grad():
        for i in range(notes_amount):
            seed = {k: v.to(device) for k, v in seed.items()}
            out = model(seed)
            pitch = torch.argmax(out["pitch"], dim=-1)
            velocity = torch.argmax(out["velocity"], dim=-1)
            duration = out["duration"].squeeze(-1).item()

            print(f"pitch: {pitch.item()}, velocity: {velocity.item()}, duration: {duration}")

            # it should start from the last note
            notes.append(pretty_midi.Note(
                start=notes[-1].end,
                end=notes[-1].end + duration,
                pitch=int(pitch.item()),
                velocity=int(velocity.item())
            ))

    return notes

