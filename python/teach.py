import numpy as np

from python.agent import DQNAgent
from python.game import *

state_size = 25  # Размер состояния агента
action_size = 4  # Количество действий агента
agent = DQNAgent(state_size, action_size)

batch_size = 32
EPISODES = 1000

for e in range(EPISODES):
    board = initialize_board()
    start = time.time()
    player = Player.PLAYER_TYPE()
    state = get_state(board)
    display = Display(BOARD_HEIGHT, BOARD_EXTENDED_WIDTH)
    total_points = 0



    #state = env.reset()  # Начальное состояние среды
    state = np.reshape(state, [1, state_size])

    for turn_number in xrange(500):
        # Move
        total_points += take_turn(board, turn_number, player)
        if not check_for_life(board):
            break
        # Reproduce
        breed(board, turn_number, REPRODUCTION_RATE)
        # Draw tiles
        for coordinate in board.get_changed_cells():
            display.draw_cell(coordinate, board)
        display.update()
        if not turn_number % int(NUMBER_OF_TURNS / 100):
            population = 0
            for c, specimens in board.specimens.items():
                population += len(specimens)
            print('{:3.0%} '.format(turn_number / NUMBER_OF_TURNS) +
                  '{:5.4}s '.format(time.time() - start) +
                  '{: 10} pts '.format(total_points) +
                  'Pop {: 5} '.format(population) +
                  'Fit ' +
                  'Avg {:7.3} '.format(TotalFitness / float(population)) +
                  'Max {: 5} '.format(MaxFitness) +
                  'AllTimeMax {: 5}'.format(AllTimeMaxFitness)
                  )









    for time in range(500):
        action = agent.act(state)





        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)