import sys
import pygame
from pygame_widgets import Slider, TextBox
import random
import numpy as np
from kalman_dim import KalmanFilter2D

NUM_POINTS = 1000
noise = 0
pause = True

pygame.init()
pygame.display.set_caption("Visual Kalman Filter")

size = width, height = 1280, 720
black = 0, 0, 0
white = 255, 255, 255
menu_background = 200, 80, 80
screen = pygame.display.set_mode(size)
menu = None
menu_description = None
noise_slider = None
output = None
min_noise_val = 0
max_noise_val = 250
min_noise_text = None
max_noise_text = None


def display_pause_menu():
    global menu, noise_slider, min_noise_text, max_noise_text, output, menu_description
    menu = pygame.draw.rect(screen, menu_background, [width/4, height/4,
                                                      width/2,
                                                      height/2], border_radius=2)
    noise_slider = Slider(screen, width*3//8, height//2, width//4, 22, min=0, max=250,
                          step=1, initial=noise, handleRadius=10)
    min_noise_text = TextBox(screen, width*3//8-40, height//2+3, 20, 30,
                             fontSize=24, colour=menu_background,
                             borderColour=menu_background)
    min_noise_text.setText(min_noise_val)
    max_noise_text = TextBox(screen, width*3//8+width//4+10, height // 2 + 3,
                             60,
                             30,
                             fontSize=24, colour=menu_background,
                             borderColour=menu_background)
    max_noise_text.setText(max_noise_val)
    output = TextBox(screen, width//2-18, height//2-32, 80, 30, fontSize=24,
                     colour=menu_background, borderColour=menu_background)
    output.setText(noise_slider.getValue())
    menu_description = TextBox(screen, width/4, height/4, width/2, height/8,
                               fontSize=40, colour=menu_background,
                               borderColour=menu_background)
    menu_description.setText("           Set the noise of the system.")
    min_noise_text.setText(min_noise_val)


'''
def calc_average(list_of_numbers):
    sum = 0
    for num in list_of_numbers:
        sum += num
    return sum / len(list_of_numbers)


def calc_covariance(a, b):
    min_length = min(len(a), len(b))
    cov = 0
    a_avg = calc_average(a)
    b_avg = calc_average(b)

    for i in range(min_length):
        cov += (a[i] - a_avg) * (b[i] - b_avg)
    return cov / min_length


def calc_var(a):
    var = 0
    a_avg = calc_average(a)
    for i in range(len(a)):
        var += (a[i] - a_avg)**2
    return var / (len(a))


def create_q():
    global x_positions, y_positions, x_velocities, y_velocities

    q = np.eye(4)
    q[0][0] = np.var(x_positions)
    q[0][1] = calc_covariance(x_positions, y_positions)
    q[0][2] = calc_covariance(x_positions, x_velocities)
    q[0][3] = calc_covariance(x_positions, y_velocities)
    q[1][0] = calc_covariance(y_positions, x_positions)
    q[1][1] = np.var(y_positions)
    q[1][2] = calc_covariance(y_positions, x_velocities)
    q[1][3] = calc_covariance(y_positions, y_velocities)
    q[2][0] = calc_covariance(x_velocities, x_positions)
    q[2][1] = calc_covariance(x_velocities, y_positions)
    q[2][2] = np.var(x_velocities)
    q[2][3] = calc_covariance(x_velocities, y_velocities)
    q[3][0] = calc_covariance(y_velocities, x_positions)
    q[3][1] = calc_covariance(y_velocities, y_positions)
    q[3][2] = calc_covariance(y_velocities, x_velocities)
    q[3][3] = np.var(y_velocities)
    return q
'''

if __name__ == "__main__":
    prev_x_pos, prev_y_pos = pygame.mouse.get_pos()
    prev_time = pygame.time.get_ticks()

    curr_x_pos = 0
    curr_y_pos = 0
    curr_time = 0

    x_positions = [0]
    y_positions = [0]

    x_velocities = [0]
    y_velocities = [0]

    count = 0

    x_pos_predictions = [0]
    y_pos_predictions = [0]

    # initialize kalman filter
    kf = KalmanFilter2D()
    display_pause_menu()
    while 1:
        pygame.time.delay(1)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    pause = not pause
                    if pause:
                        display_pause_menu()
                    else:
                        menu = None
                        noise_slider = None
                        min_noise_text = None
                        max_noise_text = None
                        output = None
                        menu_description = None

        if menu is not None:
            noise_slider.listen(events)
            noise_slider.draw()
            noise = noise_slider.getValue()
            output.setText(noise)
            output.draw()
            min_noise_text.draw()
            max_noise_text.draw()
            menu_description.draw()

        # update the position lists
        curr_x_pos, curr_y_pos = pygame.mouse.get_pos()
        x_positions.append(curr_x_pos + noise/2 - random.randint(0, noise))
        y_positions.append(curr_y_pos + noise/2 - random.randint(0, noise))
        if (len(x_positions) > NUM_POINTS or len(y_positions) > NUM_POINTS):
            x_positions.pop(0)
            y_positions.pop(0)

        # getting the current time and delta_t
        curr_time = pygame.time.get_ticks()
        delta_t = curr_time - prev_time  # to get delta_t in seconds
        prev_time = curr_time

        # don't calculate accelerations for the first few cursor inputs
        if count < 3:
            count += 1
            continue

        # update the velocity lists
        x_vel = (x_positions[-1] - x_positions[-2]) / delta_t
        y_vel = (y_positions[-1] - y_positions[-2]) / delta_t
        """print("X velocity: " + str(x_vel))
        print("Y velocity: " + str(y_vel))
        print()"""
        x_velocities.append(x_vel)
        y_velocities.append(y_vel)
        if (len(x_velocities) > NUM_POINTS or len(y_velocities) > NUM_POINTS):
            x_velocities.pop(0)
            y_velocities.pop(0)

        # implement prediction and update steps
        q = np.eye(4)
        q[0][0] = q[1][1] = 0.001
        q[2][2] = q[3][3] = 0

        #print(q / 1000)
        kf.predict(delta_t)
        state_vector = kf.update(measured_x_pos=x_positions[-1],
                                 measured_y_pos=y_positions[-1],
                                 measured_x_vel=x_velocities[-1],
                                 measured_y_vel=y_velocities[-1])

        x_pos_predictions.append(state_vector[0])
        y_pos_predictions.append(state_vector[1])

        if len(x_pos_predictions) > NUM_POINTS:
            x_pos_predictions.pop(0)
            y_pos_predictions.pop(0)

        if not pause:
            screen.fill(black)
            for i in range(0, len(x_positions)):
                curr_x = x_positions[i]
                curr_y = y_positions[i]
                pygame.draw.rect(screen, (0, 0, 255), [curr_x, curr_y, 5, 5])
            for i in range(1, len(x_pos_predictions)):
                last_x = x_pos_predictions[i - 1]
                last_y = y_pos_predictions[i - 1]
                curr_x = x_pos_predictions[i]
                curr_y = y_pos_predictions[i]
                pygame.draw.line(screen, (0, 255, 0), (last_x, last_y),
                                 (curr_x, curr_y), width=5)
            noise_text.draw()
            pygame.display.flip()
        else:
            pygame.display.update(menu)
            noise_text = TextBox(screen, width-width//8, 0, width//8, 40,
                                 fontSize=24, colour=black,
                                 borderColour=black, textColour=white)
            noise_text.setText("Noise: " + str(noise))
