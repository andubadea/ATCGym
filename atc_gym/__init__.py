from gymnasium.envs.registration import register

def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    # Register conflictart gym.
    register(
        id="ConflictArt-v0",
        entry_point="atc_gym.envs.conflict_art:ConflictArtEnv",
        max_episode_steps=500,
    )
    
    register(
        id="ConflictGenArt-v0",
        entry_point="atc_gym.envs.conflict_gen_art:ConflictGenArtEnv",
        max_episode_steps=500,
    )
    
    register(
        id="ConflictUrbanArt-v0",
        entry_point="atc_gym.envs.conflict_urban_art:ConflictUrbanArtEnv",
        max_episode_steps=1000,
    )
    
    register(
        id="ConflictSACArt-v0",
        entry_point="atc_gym.envs.conflict_SAC_art:ConflictSACArtEnv",
        max_episode_steps=1000,
    )