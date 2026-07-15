import numpy as np
import pytest

from tests.conftest import DummyModel
from utils.Results import Results
from utils.utils import FilteringCriteria


def make_results(features_test=None, target_test=None, pred_test=None, predict_score=None):
    """Build a Results instance with internal state set directly,
    bypassing predict()/set_test() (which require a real sklearn model)."""
    r = Results(DummyModel())
    r._features_test = features_test
    r._target_test = target_test
    r._pred_test = pred_test
    r._predict_score = predict_score
    return r


# maps the kwarg name used in make_results() -> the private attribute name on Results
ATTR_MAP = {
    "features_test": "_features_test",
    "target_test": "_target_test",
    "pred_test": "_pred_test",
    "predict_score": "_predict_score",
}


class TestAppendResultsReturnValue:
    def test_returns_self(self):
        r1 = make_results()
        r2 = make_results()
        assert r1.append_results(r2) is r1


class TestAppendResultsAllNone:
    def test_all_fields_remain_none_when_both_sides_empty(self):
        r1 = make_results()
        r2 = make_results()
        r1.append_results(r2)
        assert r1._features_test is None
        assert r1._target_test is None
        assert r1._pred_test is None
        assert r1._predict_score is None


@pytest.mark.parametrize("field", list(ATTR_MAP.keys()))
class TestAppendResultsPerField:
    """Exercises the 3 branches of a single if/else block in isolation,
    keeping every other field at None so branches don't interact."""

    def test_skip_when_incoming_field_is_none(self, field):
        existing = np.array([[1, 2], [3, 4]])
        r1 = make_results(**{field: existing})
        r2 = make_results()  # nothing set on the incoming side
        r1.append_results(r2)
        np.testing.assert_array_equal(getattr(r1, ATTR_MAP[field]), existing)

    def test_copy_when_self_field_is_none(self, field):
        incoming = np.array([[5, 6], [7, 8]])
        r1 = make_results()
        r2 = make_results(**{field: incoming})
        r1.append_results(r2)
        merged = getattr(r1, ATTR_MAP[field])
        np.testing.assert_array_equal(merged, incoming)
        assert merged is not incoming  # must be a .copy(), not the same object

    def test_concatenate_when_both_present(self, field):
        existing = np.array([[1, 2], [3, 4]])
        incoming = np.array([[5, 6], [7, 8]])
        r1 = make_results(**{field: existing})
        r2 = make_results(**{field: incoming})
        r1.append_results(r2)
        expected = np.concatenate([existing, incoming], axis=0)
        np.testing.assert_array_equal(getattr(r1, ATTR_MAP[field]), expected)

    def test_copy_is_independent_of_source(self, field):
        """Mutating the merged array on r1 must not affect r2's original array."""
        incoming = np.array([[5, 6], [7, 8]])
        r1 = make_results()
        r2 = make_results(**{field: incoming})
        r1.append_results(r2)
        getattr(r1, ATTR_MAP[field])[0, 0] = 999
        assert incoming[0, 0] == 5

    def test_does_not_mutate_incoming_result(self, field):
        """The `result` argument passed to append_results must be left untouched."""
        existing = np.array([[1, 2], [3, 4]])
        incoming_original = np.array([[5, 6], [7, 8]])
        incoming = incoming_original.copy()
        r1 = make_results(**{field: existing})
        r2 = make_results(**{field: incoming})
        r1.append_results(r2)
        np.testing.assert_array_equal(getattr(r2, ATTR_MAP[field]), incoming_original)


class TestAppendResultsFullMerge:
    """Sanity checks with all four fields populated together, and chaining."""

    def test_merges_all_fields_simultaneously(self):
        r1 = make_results(
            features_test=np.array([[1, 1]]),
            target_test=np.array([0]),
            pred_test=np.array([0]),
            predict_score=np.array([[0.9, 0.1]]),
        )
        r2 = make_results(
            features_test=np.array([[2, 2]]),
            target_test=np.array([1]),
            pred_test=np.array([1]),
            predict_score=np.array([[0.2, 0.8]]),
        )

        r1.append_results(r2)

        np.testing.assert_array_equal(r1._features_test, np.array([[1, 1], [2, 2]]))
        np.testing.assert_array_equal(r1._target_test, np.array([0, 1]))
        np.testing.assert_array_equal(r1._pred_test, np.array([0, 1]))
        np.testing.assert_array_equal(
            r1._predict_score, np.array([[0.9, 0.1], [0.2, 0.8]])
        )

    def test_mixed_fields_some_none_some_set(self):
        """One field only exists on r2, another only on r1 -> both branches
        (copy vs skip) fire within the same call."""
        r1 = make_results(features_test=np.array([[1, 1]]), target_test=None)
        r2 = make_results(features_test=None, target_test=np.array([1]))

        r1.append_results(r2)

        np.testing.assert_array_equal(r1._features_test, np.array([[1, 1]]))
        np.testing.assert_array_equal(r1._target_test, np.array([1]))

    def test_multiple_sequential_appends_accumulate(self):
        r1 = make_results(features_test=np.array([[1, 1]]))
        r2 = make_results(features_test=np.array([[2, 2]]))
        r3 = make_results(features_test=np.array([[3, 3]]))

        r1.append_results(r2).append_results(r3)

        np.testing.assert_array_equal(
            r1._features_test, np.array([[1, 1], [2, 2], [3, 3]])
        )
def test_filter_by_returns_best(classifier):
    r1 = Results(None)
    r1._accuracy = 0.7

    r2 = Results(None)
    r2._accuracy = 0.9

    classifier._results = [r1, r2]

    result = classifier.filter_by(FilteringCriteria.accuracy)

    assert result == [r2]


def test_results_property(classifier):
    classifier._results = [1, 2, 3]
    assert classifier.results == [1, 2, 3]
