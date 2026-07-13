from types import SimpleNamespace

import h5py
import numpy as np

from pimsr_inversion.train3d import Volume3DDataset, train


def _sample(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("observations/apparent_resistivity", data=np.full((2, 2, 4, 4), 100.0, dtype="f4"))
        f.create_dataset("observations/phase", data=np.full((2, 2, 4, 4), 45.0, dtype="f4"))
        f.create_dataset("target/log10_resistivity", data=np.full((4, 4, 4), 2.0, dtype="f4"))


def test_volume_dataset_shape(tmp_path):
    _sample(tmp_path / "sample.h5")
    obs, target = Volume3DDataset(tmp_path)[0]
    assert obs.shape == (4, 2, 4, 4)
    assert target.shape == (4, 4, 4)


def test_cpu_smoke_training_is_resumable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _sample(data / "sample.h5")
    out = tmp_path / "out"
    args = SimpleNamespace(
        data=str(data), out=str(out), preset="local-8gb", epochs=1,
        lr=2e-4, workers=0, resume=None,
    )
    result = train(args)
    assert len(result["history"]) == 1
    assert (out / "last3d.pt").exists()
    assert (out / "history3d.json").exists()
