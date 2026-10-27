"""Check runtime ember parameter object seen by the generator (Phase 2H Step 4)."""

from wf052.surrogate.standin_generator import (
    get_frozen_teacher_v2_config_path,
    load_and_verify_frozen_teacher_v2,
)


def main() -> None:
    p = get_frozen_teacher_v2_config_path()
    cfg, file_sha, sim_hash, t_params, e_params = load_and_verify_frozen_teacher_v2(p)

    print("=" * 60)
    print("PHASE 2H STEP 4: RUNTIME EMBER PARAMETERS")
    print("=" * 60)
    print(f"Config path:        {p}")
    print(f"File SHA-256:       {file_sha}")
    print(f"Canonical Sim Hash: {sim_hash}")
    print(f"EmberParams object: {repr(e_params)}")
    print(f"p_urban_wui:        {e_params.p_urban_wui}")
    print("=" * 60)


if __name__ == "__main__":
    main()
