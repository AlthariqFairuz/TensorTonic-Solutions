def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    new_val = []
    for state in range(len(values)):
        action_val = []
        for action in range(len(transitions[state])):
            expected_next = sum(
                probability * next_value
                for probability, next_value in zip(transitions[state][action], values)
            )
            action_val.append(rewards[state][action] + gamma * expected_next)
        new_val.append(max(action_val))
    return new_val