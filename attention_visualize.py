import pygame
import numpy as np


def get_color(num):
    if num % 2 == 0:
        half_num = int(num / 2)
        color_list = np.linspace(start=10, stop=250, endpoint=False, num=half_num, dtype=np.int)
        color = []
        for i in range(half_num):
            color.append((int(255 - color_list[i]), color_list[i], 0))
        for i in range(half_num):
            color.append((0, int(255 - color_list[i]), color_list[i]))
        return color
    else:
        half_num = int((num - 1) / 2)
        color_list = np.linspace(start=0, stop=255, endpoint=False, num=half_num, dtype=np.int)
        color = []
        for i in range(half_num):
            color.append((int(255 - color_list[i]), color_list[i], 0))
        color.append((0, 255, 0))
        for i in range(half_num):
            color.append((0, int(255 - color_list[i]), color_list[i]))
        return color


def attention_out_pic(words, attention_array, name='word', font_size=50):
    assert len(words) == len(attention_array)

    pygame.init()
    font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)  # one letter is 15 height and 9.5 width
    text = font.render('w', True, (0, 0, 0))
    letter_width, letter_height = text.get_size()

    letter_width = int(letter_width * 0.7)
    letter_height = int(letter_height * 1.2)
    front = 5.

    word_num = len(words)
    color_set = get_color(word_num)
    letter_length = word_num
    for word in words:
        letter_length += len(word)

    window = pygame.display.set_mode((int(letter_length * letter_width + 2 * front), letter_height))
    window.fill((255, 255, 255))

    # sort
    order = sorted(enumerate(attention_array), key=lambda x: x[1], reverse=True)
    position = front
    for i in range(word_num):
        text = font.render(words[i], True, color_set[order[i][0]])
        width, _ = text.get_size()
        window.blit(text, (position, letter_height * 0.1))
        position += width + 0.8 * letter_width

    pygame.image.save(window, name + ".png")


lst = [chr(ord('a')+i) for i in range(26)]
attention_out_pic(lst, np.linspace(0, 26, num=26), font_size=80, name='color_map')
