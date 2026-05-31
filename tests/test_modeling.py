import numpy as np

from entrance_counter.modeling import extract_track_outputs


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = FakeTensor(np.array([[1.0, 2.0, 11.0, 22.0]], dtype=float))
        self.id = FakeTensor(np.array([42], dtype=float))
        self.conf = FakeTensor(np.array([0.87], dtype=float))


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


def test_extract_track_outputs_returns_empty_arrays_when_no_boxes() -> None:
    boxes, ids, confs = extract_track_outputs(FakeResult(boxes=None))

    assert boxes.shape == (0, 4)
    assert ids.shape == (0,)
    assert confs.shape == (0,)


def test_extract_track_outputs_returns_empty_arrays_when_no_track_ids() -> None:
    fake_boxes = FakeBoxes()
    fake_boxes.id = None

    boxes, ids, confs = extract_track_outputs(FakeResult(boxes=fake_boxes))

    assert boxes.shape == (0, 4)
    assert ids.shape == (0,)
    assert confs.shape == (0,)


def test_extract_track_outputs_converts_fake_ultralytics_tensors() -> None:
    boxes, ids, confs = extract_track_outputs(FakeResult(boxes=FakeBoxes()))

    np.testing.assert_allclose(boxes, np.array([[1.0, 2.0, 11.0, 22.0]], dtype=float))
    np.testing.assert_array_equal(ids, np.array([42], dtype=int))
    np.testing.assert_allclose(confs, np.array([0.87], dtype=float))
