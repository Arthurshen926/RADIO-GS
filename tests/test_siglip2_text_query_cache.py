from radio_gs.scripts.build_siglip2_text_query_cache import _queries


def test_long_comma_separated_query_bank_is_not_treated_as_a_path() -> None:
    values = [f"category {index}" for index in range(80)]

    assert _queries(",".join(values)) == values
