import hashlib


def test_inventory_records_selected_cached_weights(tmp_path, monkeypatch):
    import torch
    from music_re.core.provenance import model_inventory
    checkpoint_dir = tmp_path / 'hub' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / '955717e8-fixture.th').write_bytes(b'weights')
    (checkpoint_dir / 'unrelated.th').write_bytes(b'not used')
    monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path / 'hub'))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache', lambda *args: None)
    inventory = model_inventory('htdemucs', None)
    assert inventory['separator'][0]['sha256'] == hashlib.sha256(b'weights').hexdigest()
    assert inventory['separator'][0]['id'] == '955717e8-fixture.th'
    assert len(inventory['separator']) == 1
    assert inventory['tonal'][0]['sha256']
