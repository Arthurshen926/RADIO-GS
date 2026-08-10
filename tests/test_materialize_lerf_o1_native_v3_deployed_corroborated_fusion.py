from radio_gs.querying import (
    deployed_scale_corroborated_native_v3_support_fusion as deployed,
)
from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_deployed_corroborated_fusion as materializer,
)


def test_deployed_materializer_binds_parameter_free_contract() -> None:
    assert materializer.EXECUTION_SCHEMA.endswith(".v1")
    assert deployed.readout_contract()["invariants"]["new_tunable_constants"] is False
