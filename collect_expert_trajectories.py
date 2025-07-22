import pickle
from surrol.tasks.needle_pick import NeedlePick

def collect_expert_trajectories(num_episodes=100, max_steps=100, save_path="expert_trajectories.pkl"):
    env = NeedlePick(render_mode=None)  # No GUI for faster collection
    trajectories = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }
        episode_success = False
        for step in range(max_steps):
            action = env.get_oracle_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)
            episode_data['infos'].append(info)
            
            # Check for success at this step
            if info.get('is_success', False):
                episode_success = True
            
            obs = next_obs
            if done:
                break
        trajectories.append(episode_data)
        print(f"Episode {episode+1}/{num_episodes} finished after {step+1} steps. Success: {episode_success}")

    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved {num_episodes} expert trajectories to {save_path}")

if __name__ == "__main__":
    collect_expert_trajectories(num_episodes=100, max_steps=100, save_path="expert_trajectories.pkl")
