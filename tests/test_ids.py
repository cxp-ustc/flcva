from flcva.ids import payload_sha256, split_bucket, stable_id, stable_seed


def test_stable_identity_functions() -> None:
    assert stable_id("x", "a", 1) == stable_id("x", "a", 1)
    assert stable_id("x", "a", 1) != stable_id("x", "a", 2)
    assert stable_seed(17, "a") == stable_seed(17, "a")
    assert 0 <= split_bucket("gsm8k", "item") < 100
    assert payload_sha256({"b": 2, "a": 1}) == payload_sha256(
        {"a": 1, "b": 2}
    )

