from npov_drift.ingest.article_type import (
    CONTESTED_PRIOR,
    classify_article_type,
    map_topics_to_bucket,
)


def test_politics_bucket():
    b, p = map_topics_to_bucket(
        {"History and Society.Politics and government": 0.8, "STEM.Physics": 0.1}
    )
    assert b == "politics"
    assert p == CONTESTED_PRIOR["politics"]


def test_medicine_matched_before_generic_stem():
    b, _ = map_topics_to_bucket({"STEM.Medicine & Health": 0.7, "STEM.Biology": 0.2})
    assert b == "medicine"


def test_geography_and_biography():
    # The real geography signal is "Geographical", not the locational Regions.*.
    assert map_topics_to_bucket({"Geography.Geographical": 0.9})[0] == "geography"
    assert map_topics_to_bucket({"Culture.Biography.Biography": 0.95})[0] == "biography"


def test_region_tags_do_not_force_geography():
    # Regression: "Capital punishment" tops out on a Regions.Asia tag because it
    # is discussed across world regions, but it is a politics/society topic.
    scores = {
        "Geography.Regions.Asia.Asia*": 0.747,
        "History and Society.Politics and government": 0.723,
        "History and Society.Society": 0.687,
        "Geography.Regions.Africa.Africa*": 0.128,
        "Geography.Regions.Europe.Europe*": 0.105,
    }
    bucket, prior = map_topics_to_bucket(scores)
    assert bucket == "politics"
    assert prior == CONTESTED_PRIOR["politics"]


def test_empty_scores_unknown():
    b, p = map_topics_to_bucket({})
    assert b == "unknown"
    assert p == CONTESTED_PRIOR["unknown"]


def test_contested_prior_monotonicity():
    # Sanity: clearly-contested types rank above clearly-factual ones.
    assert CONTESTED_PRIOR["politics"] > CONTESTED_PRIOR["science"]
    assert CONTESTED_PRIOR["religion"] > CONTESTED_PRIOR["geography"]


def test_classify_via_ores_response(client_cls):
    ores = {
        "enwiki": {
            "scores": {
                "123": {
                    "articletopic": {
                        "score": {
                            "probability": {
                                "History and Society.Politics and government": 0.9,
                                "STEM.Physics": 0.05,
                            }
                        }
                    }
                }
            }
        }
    }
    c = client_cls(external_responses=[ores])
    at = classify_article_type(c, 123)
    assert at.bucket == "politics"
    assert at.method == "ores"
    assert at.scores["History and Society.Politics and government"] == 0.9


def test_classify_falls_back_to_unavailable(client_cls):
    # ORES returns an unusable payload, Lift Wing too -> "unavailable".
    c = client_cls(external_responses=[{"junk": 1}, {"also": "junk"}])
    at = classify_article_type(c, 999)
    assert at.method == "unavailable"
    assert at.bucket == "unknown"
