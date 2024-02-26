from gymnasium.envs.registration import register

def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    # Register conflictart gym
    register(
        id="ConflictArt-v0",
        entry_point="bluesky_gym.envs.conflict_art:ConflictArtEnv",
        max_episode_steps=300,
    )
    
    register(
        id="ConflictArtRel-v0",
        entry_point="bluesky_gym.envs.conflict_art:ConflictArtRelEnv",
        max_episode_steps=300,
    )