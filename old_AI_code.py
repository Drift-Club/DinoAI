# Gère la partie en cours
def nouvelle_partie(display_option, speed, params):
    # On crée l'agent de notre IA
    agent = DQNAgent(params)

    # S'il existe des poids (ie on a déjà fait tourner l'IA, alors on les charge)
    weights_filepath = params['weights_path']
    if params['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    nb_jeux_joues = 0
    score_plot = []
    counter_plot = []
    record = 0

    # On fait jouer notre IA autant de parties que requis
    while nb_jeux_joues < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialisation de la partie en cours
        game = Game(440, 440)
        player1 = game.player
        ennemis1 = game.ennemis

        # On effectue la première action de l'IA
        initialiser_partie(player1, game, ennemis1, agent, params['batch_size'])
        if display_option:
            display(player1, ennemis1, game, record)

        # Tant que l'agent n'a pas perdu
        while not game.crash:
            if not params['train']:  # Pas d'aléatoire si on entraîne pas l'IA
                agent.epsilon = 0
            else:
                # La fonction agent.epsilon détermine le caractère aléatoire des actions
                agent.epsilon = 1 - (nb_jeux_joues * params['epsilon_decay_linear'])

            # Récupère l'état précédent
            state_old = agent.get_state(game, player1, ennemis1)

            # Soit on performe une action au hasard si on sait rien faire,
            # sinon on prend une action en fonction des connaissances de l'IA
            if randint(0, 1) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # On décide de l'action en fonction de l'état précédent
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # On effectue la nouvelle action
            player1.do_move(final_move, player1.x, player1.y, game, ennemis1, agent)
            state_new = agent.get_state(game, player1, ennemis1)

            # On calcule le reward, si le joueur a perdu ou continue de survivre
            reward = agent.set_reward(player1, game.crash)

            if params['train']:
                # On stocke cette action dans la mémoire à court terme
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # On stocke cette action dans la mémoire à long terme
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            # On enregistre le high score
            record = high_score
            if display_option:
                display(player1, ennemis1, game, record)
                pygame.time.wait(speed)

        # La partie est terminée, on en tire toutes les conclusions ...
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])

        # Fin d'une partie, on affiche le record actuel
        nb_jeux_joues += 1
        print(f'Partie n° {nb_jeux_joues}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(nb_jeux_joues)
    if params['train']:
        agent.model.save_weights(params['weights_path'])