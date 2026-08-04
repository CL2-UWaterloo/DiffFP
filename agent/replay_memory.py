import numpy as np
import torch


class ReplayMemory():
    """Buffer to store environment transitions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.masks = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def append(self, state, action, reward, next_state, mask):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.masks[self.idx], mask)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def append_batch(self, states, actions, rewards, next_states, masks):
        """
        Append a batch of transitions to the replay memory.

        Args:
            states: NumPy array of shape (batch_size, state_dim)
            actions: NumPy array of shape (batch_size, action_dim)
            rewards: NumPy array of shape (batch_size,) or shape (batch_size, ...) if multi-dimensional
            next_states: NumPy array of shape (batch_size, state_dim)
            masks: NumPy array of shape (batch_size,) or matching the rewards shape.
        """
        batch_size = states.shape[0]
        capacity = self.capacity
        idx = self.idx

        # Check if the batch will wrap around the end of the buffer.
        if idx + batch_size <= capacity:
            self.states[idx:idx+batch_size] = states
            self.actions[idx:idx+batch_size] = actions
            self.rewards[idx:idx+batch_size][:,0] = rewards
            self.next_states[idx:idx+batch_size] = next_states
            self.masks[idx:idx+batch_size][:,0] = masks
            self.idx = (idx + batch_size) % capacity
            if self.idx == 0:
                self.full = True
        else:
            # Split the batch into two parts.
            first_part = capacity - idx
            second_part = batch_size - first_part

            self.states[idx:] = states[:first_part]
            self.actions[idx:] = actions[:first_part]
            self.rewards[idx:][:,0] = rewards[:first_part]
            self.next_states[idx:] = next_states[:first_part]
            self.masks[idx:][:,0] = masks[:first_part]

            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part][:,0] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.masks[:second_part][:,0] = masks[first_part:]
            self.idx = second_part
            self.full = True

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        states = torch.as_tensor(self.states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device)
        masks = torch.as_tensor(self.masks[idxs], device=self.device)

        return states, actions, rewards, next_states, masks


class DiffusionMemory():
    """Buffer to store best actions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.best_actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)

        self.idx = 0
        self.full = False

    def append(self, state, action):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.best_actions[self.idx], action)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def append_batch(self, states, actions):
        """
        Append a batch of (state, action) samples to the replay memory.
        
        Args:
            states: A NumPy array of shape (batch_size, ...) matching the shape of a single state.
            actions: A NumPy array of shape (batch_size, ...) matching the shape of a single action.
        """
        batch_size = states.shape[0]
        capacity = self.capacity
        idx = self.idx

        # If the entire batch fits without wrapping around:
        if idx + batch_size <= capacity:
            self.states[idx:idx+batch_size] = states
            self.best_actions[idx:idx+batch_size] = actions
            self.idx = (idx + batch_size) % capacity
            if self.idx == 0:
                self.full = True
        else:
            # Split the batch into two parts:
            first_part = capacity - idx
            second_part = batch_size - first_part

            self.states[idx:] = states[:first_part]
            self.best_actions[idx:] = actions[:first_part]

            self.states[:second_part] = states[first_part:]
            self.best_actions[:second_part] = actions[first_part:]

            self.idx = second_part
            self.full = True



    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        states = torch.as_tensor(self.states[idxs], device=self.device)
        best_actions = torch.as_tensor(self.best_actions[idxs], device=self.device)

        best_actions.requires_grad_(True)

        return states, best_actions, idxs

    def replace(self, idxs, best_actions):
        np.copyto(self.best_actions[idxs], best_actions)
    