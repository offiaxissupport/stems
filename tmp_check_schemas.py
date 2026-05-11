from citylearn.citylearn import CityLearnEnv
schemas = [
    'citylearn_challenge_2022_phase_all',
    'citylearn_challenge_2023_phase_2_online_evaluation_1',
    'citylearn_challenge_2023_phase_2_online_evaluation_2',
    'tx_travis_county_neighborhood',
    'ca_alameda_county_neighborhood',
]
for s in schemas:
    try:
        e = CityLearnEnv(s, central_agent=False)
        print(f"{s} -> {len(e.observation_space)} buildings, obs_dim={e.observation_space[0].shape[0]}")
    except Exception as ex:
        print(f"{s} -> ERROR: {ex}")
