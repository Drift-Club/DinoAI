import os
import pygame
import random
import numpy as np
import sys
from DQN import DQNAgent
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical  # bibliothèque Keras permettant un dvp IA de haut niveau

#######################################
# Variables globales pour le jeu (ne concerne pas l'IA)
#######################################
pygame.mixer.pre_init(44100, -16, 2, 2048)  # fix audio delay
pygame.init()

scr_size = (width, height) = (600, 150)
FPS = 60
gravity = 0.65

black = (0, 0, 0)
white = (255, 255, 255)
background_col = (235, 235, 235)
RLEACCEL = 16384

high_score = 0

screen = pygame.display.set_mode(scr_size)
clock = pygame.time.Clock()
pygame.display.set_caption("chrome://dino")

jump_sound = pygame.mixer.Sound('chrome_dino/sprites/jump.wav')
die_sound = pygame.mixer.Sound('chrome_dino/sprites/die.wav')
checkPoint_sound = pygame.mixer.Sound('chrome_dino/sprites/checkPoint.wav')


#######################################
#           Code de l'IA
#######################################
#   On définit les paramètres de l'IA manuellement
def définir_paramètres():
    params = dict()
    params['epsilon_decay_linear'] = 1 / 80  # La fonction agent.epsilon détermine le caractère aléatoire des actions
    params['learning_rate'] = 0.001
    params['first_layer_size'] = 150  # neurons dans la première couche
    params['second_layer_size'] = 150  # dans la deuxième
    params['third_layer_size'] = 150  # dans la troisième
    params['episodes'] = 150  # Nombre de parties à jouer pour entraîner l'IA
    params['memory_size'] = 2500  # Taille de la mémoire
    params['batch_size'] = 500  # 500 de base (ceci est un test)
    params['weights_path'] = 'weights/weights.hdf5'  # endroit de stockages des poids (weights)
    params[
        'load_weights'] = False  # Charger les poids pré-calculés (regarder l'IA jouer avec ses connaissances ultérieures)
    params['train'] = True  # Entraîner l'IA, ne pas utiliser les poids
    return params


#################################
# Contenu du jeu qui n'a pas changé modifié pour être joué par notre agent
#################################
def load_image(
        name,
        sizex=-1,
        sizey=-1,
        color_key=None,
):
    fullname = os.path.join('chrome_dino/sprites', name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if color_key is not None:
        if color_key == -1:
            color_key = image.get_at((0, 0))
        image.set_colorkey(color_key, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return image, image.get_rect()


def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex=-1,
        scaley=-1,
        colorkey=None,
):
    fullname = os.path.join('chrome_dino/sprites', sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width / nx
    sizey = sheet_rect.height / ny

    for i in range(0, ny):
        for j in range(0, nx):
            rect = pygame.Rect((j * sizex, i * sizey, sizex, sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet, (0, 0), rect)

            if colorkey is not None:
                if colorkey == -1:
                    colorkey = image.get_at((0, 0))
                image.set_colorkey(colorkey, RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image, (scalex, scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites, sprite_rect


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        color="b",
        x_jitter=.1,
        line_kws={'color': 'green'}
    )
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def disp_gameOver_msg(retbutton_image, gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = int(width / 2)
    retbutton_rect.top = int(height * 0.52)

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = int(width / 2)
    gameover_rect.centery = int(height * 0.35)

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)


def extractDigits(number):
    if number > -1:
        digits = []
        while number / 10 != 0:
            digits.append(number % 10)
            number = int(number / 10)

        digits.append(number % 10)
        for i in range(len(digits), 5):
            digits.append(0)
        digits.reverse()
        return digits


class Dino:
    def __init__(self, sizex=-1, sizey=-1):
        self.images, self.rect = load_sprite_sheet('dino.png', 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet('dino_ducking.png', 2, 1, 59, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width / 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image, self.rect)

    def check_bounds(self):
        if self.rect.bottom > int(0.98 * height):
            self.rect.bottom = int(0.98 * height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2

        if self.isDead:
            self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index) % 2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.check_bounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pygame.mixer.get_init() is not None:
                    checkPoint_sound.play()

        self.counter = (self.counter + 1)


class Cactus(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('cacti-small.png', 3, 1, sizex, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width + self.rect.width
        self.image = self.images[random.randrange(0, 3)]
        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()


class Ptera(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('ptera.png', 2, 1, sizex, sizey, -1)
        self.ptera_height = [height * 0.82, height * 0.75, height * 0.60]
        self.rect.centery = self.ptera_height[random.randrange(0, 3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()


class Ground():
    def __init__(self, speed=-5):
        self.image, self.rect = load_image('ground.png', -1, -1, -1)
        self.image1, self.rect1 = load_image('ground.png', -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image('cloud.png', int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard():
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.tempimages, self.temprect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.image = pygame.Surface((55, int(11 * 6 / 5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width * 0.89
        else:
            self.rect.left = int(x)
        if y == -1:
            self.rect.top = height * 0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self, score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


def lancer_IA():
    # On crée l'agent de notre IA
    params = définir_paramètres()
    agent = DQNAgent(params)

    # Prend un argument pour savoir si on affiche ou pas l'écran (gagne de la vitesse)
    display_game = sys.argv.pop()
    if display_game is None:
        display_game = True

    # S'il existe des poids (ie on a déjà fait tourner l'IA, alors on les charge)
    weights_filepath = params['weights_path']
    if params['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    nb_jeux_joues = 0
    score_plot = []
    counter_plot = []
    record = 0
    avg_scores_aleatoires = 20
    avg_scores_memoire = 20

    # On fait jouer notre IA autant de parties que requis
    while nb_jeux_joues < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialisation de la partie en cours
        global high_score
        gamespeed = 4
        startMenu = False
        gameOver = False
        gameQuit = False
        playerDino = Dino(44, 47)
        new_ground = Ground(-1 * gamespeed)
        scb = Scoreboard()
        highsc = Scoreboard(width * 0.78)
        counter = 0

        cacti = pygame.sprite.Group()
        pteras = pygame.sprite.Group()
        clouds = pygame.sprite.Group()
        last_obstacle = pygame.sprite.Group()

        Cactus.containers = cacti
        Ptera.containers = pteras
        Cloud.containers = clouds

        retbutton_image, retbutton_rect = load_image('replay_button.png', 35, 31, -1)
        gameover_image, gameover_rect = load_image('game_over.png', 190, 11, -1)

        temp_images, temp_rect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        HI_image = pygame.Surface((22, int(11 * 6 / 5)))
        HI_rect = HI_image.get_rect()
        HI_image.fill(background_col)
        HI_image.blit(temp_images[10], temp_rect)
        temp_rect.left += temp_rect.width
        HI_image.blit(temp_images[11], temp_rect)
        HI_rect.top = height * 0.1
        HI_rect.left = width * 0.73

        # Lancement de la partie
        if not gameQuit:
            while startMenu:
                pass
            while not gameOver:
                if pygame.display.get_surface() is None:
                    print("Couldn't load display surface 2")
                    gameQuit = True
                    gameOver = True
                else:
                    # TODO DONE Calcul du epsilon
                    if not params['train']:  # Pas d'aléatoire si on entraîne pas l'IA
                        agent.epsilon = 0
                    else:
                        # La fonction agent.epsilon détermine le caractère aléatoire des actions
                        agent.epsilon = 1 - (nb_jeux_joues * params['epsilon_decay_linear'])

                    # TODO DONE Récupère l'état
                    state_old = agent.get_state(playerDino, cacti, pteras)

                    # TODO DONE Soit on performe une action au hasard si on sait rien faire,
                    # TODO DONE sinon on prend une action en fonction des connaissances de l'IA (retirer events)
                    if randint(0, 1) < agent.epsilon:
                        move = to_categorical(randint(0, 2), num_classes=3)
                    else:
                        # On décide de l'action en fonction de l'état précédent
                        prediction = agent.model.predict(state_old.reshape((1, 8)))
                        move = to_categorical(np.argmax(prediction[0]), num_classes=3)
                        # Le final move / action est un array [rester_droit sauter saccroupir]

                    # TODO DONE on effectue l'action
                    if np.array_equal(move, [1, 0, 0]):
                        # print("TOUT DROIT")
                        playerDino.isDucking = False
                        # On avance tt droit, si le dino est accroupi il se relève
                    elif np.array_equal(move, [0, 1, 0]):
                        # print("SAUTE")
                        if playerDino.rect.bottom == int(0.98 * height):
                            playerDino.isJumping = True
                            # if pygame.mixer.get_init() is not None:
                            # jump_sound.play()
                            playerDino.movement[1] = -1 * playerDino.jumpSpeed
                    elif np.array_equal(move, [0, 0, 1]):
                        # print("S'ACCROUPIR")
                        if not (playerDino.isJumping and playerDino.isDead):
                            playerDino.isDucking = True

                    # Mouvement des ennemis et détection des collisions
                    for c in cacti:
                        c.movement[0] = -1 * gamespeed
                        if pygame.sprite.collide_mask(playerDino, c):
                            playerDino.isDead = True
                            # if pygame.mixer.get_init() is not None:
                            #   die_sound.play()

                    for p in pteras:
                        p.movement[0] = -1 * gamespeed
                        if pygame.sprite.collide_mask(playerDino, p):
                            playerDino.isDead = True
                            # if pygame.mixer.get_init() is not None:
                            #   die_sound.play()

                    # TODO DONE Calcul du reward et du nouveau state
                    state_new = agent.get_state(playerDino, cacti, pteras)
                    reward = agent.set_reward(playerDino, cacti, pteras)

                    # TODO DONE Enregistrement dans la mémoire
                    if params['train']:
                        # On stocke cette action dans la mémoire à court terme
                        agent.train_short_memory(state_old, move, reward, state_new, playerDino.isDead)
                        # On stocke cette action dans la mémoire à long terme
                        agent.remember(state_old, move, reward, state_new, playerDino.isDead)

                    # Génération de nouveaux ennemis
                    if len(cacti) < 2:
                        if len(cacti) == 0:
                            last_obstacle.empty()
                            last_obstacle.add(Cactus(gamespeed, 40, 40))
                        else:
                            for l in last_obstacle:
                                if l.rect.right < width * 0.7 and random.randrange(0, 50) == 10:
                                    last_obstacle.empty()
                                    last_obstacle.add(Cactus(gamespeed, 40, 40))

                    if len(pteras) == 0 and random.randrange(0, 200) == 10 and counter > 500:
                        for l in last_obstacle:
                            if l.rect.right < width * 0.8:
                                last_obstacle.empty()
                                last_obstacle.add(Ptera(gamespeed, 46, 40))

                    if len(clouds) < 5 and random.randrange(0, 300) == 10:
                        Cloud(width, random.randrange(height / 5, height / 2))

                    # Màjs graphiques
                    playerDino.update()
                    cacti.update()
                    pteras.update()
                    clouds.update()
                    new_ground.update()
                    scb.update(playerDino.score)
                    highsc.update(high_score)

                    if pygame.display.get_surface() is not None:
                        screen.fill(background_col)
                        new_ground.draw()
                        clouds.draw(screen)
                        scb.draw()
                        if high_score != 0:
                            highsc.draw()
                            screen.blit(HI_image, HI_rect)
                        cacti.draw(screen)
                        pteras.draw(screen)
                        playerDino.draw()

                        pygame.display.update()
                    clock.tick(FPS)

                    # Mort du dino, récupérer high score
                    if playerDino.isDead:
                        gameOver = True
                        if playerDino.score > high_score:
                            high_score = playerDino.score
                            record = playerDino.score

                    if counter % 700 == 699:
                        new_ground.speed -= 1
                        gamespeed += 1

                    counter = (counter + 1)

                if gameQuit:
                    break

            # Déclenchée pour le game over
            if gameOver:
                if pygame.display.get_surface() is None:
                    print("Couldn't load display surface 3")
                    gameQuit = True
                    gameOver = False
                else:
                    # TODO DONE Le jeu est terminé, on tire les conséquences
                    if params['train']:
                        agent.replay_new(agent.memory, params['batch_size'])

                highsc.update(high_score)
                if pygame.display.get_surface() is not None:
                    disp_gameOver_msg(retbutton_image, gameover_image)
                    if high_score != 0:
                        highsc.draw()
                        screen.blit(HI_image, HI_rect)
                    pygame.display.update()

                # TODO DONE fin d'une partie, on affiche le record et enregistre les poids
                nb_jeux_joues += 1
                print("**************************************************************")
                print(f'Partie n° {nb_jeux_joues}      Score: {playerDino.score}')
                score_plot.append(playerDino.score)
                counter_plot.append(nb_jeux_joues)
                if nb_jeux_joues * params['epsilon_decay_linear'] >= 1:
                    print("Utilisation de la mémoire uniquement")
                    avg_scores_memoire = (avg_scores_memoire + playerDino.score) / 2
                else:
                    avg_scores_aleatoires = (avg_scores_aleatoires + playerDino.score) / 2

                clock.tick(FPS)

    if params['train']:
        agent.model.save_weights(params['weights_path'])
    # plot_seaborn(counter_plot, score_plot)
    print("***************************************************")
    print("Fin de l'entraînement de cette IA.")
    print("Liste des scores : ")
    print(*score_plot)
    print("Record : " + str(record))
    print("Entraînée sur " + str(params["episodes"]) + " épisodes")
    print("Moyenne aléatoire : " + str(avg_scores_aleatoires))
    print("Moyenne supervisée : " + str(avg_scores_memoire))
    print("***************************************************")


lancer_IA()
