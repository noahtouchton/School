import pygame

pygame.init()

screen = pygame.display.set_mode((800, 600))

clock = pygame.time.Clock()

while True:
    #Process player inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
            raise SystemExit
        
    # Do logical updates here

    screen.fill("purple")

    # Render the graphics here

    pygame.display.flip()
    clock.tick(60)

