class EpisodeBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, episode):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the oldest episode
        self.buffer.append(episode)

    def sample(self, num_samples):
        import random
        return random.sample(self.buffer, min(num_samples, len(self.buffer)))

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"EpisodeBuffer(capacity={self.capacity}, size={len(self)})"