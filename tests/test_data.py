from flowns.data import load_recommender_data


def test_load_atomic_interactions_and_leave_one_out_split():
    data = load_recommender_data(
        {
            "data": {
                "inter_file": "tests/fixtures/ml-mini/ml-mini.inter",
            }
        }
    )

    assert data.num_users == 3
    assert data.num_items == 6
    assert len(data.train_pairs) == 5
    assert data.valid_targets[0] == 2
    assert data.test_targets[0] == 3
    assert len(data.seq_train_samples) == 2
    assert len(data.seq_valid_samples) == 3
    assert len(data.seq_test_samples) == 3
