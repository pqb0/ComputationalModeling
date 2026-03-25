import math
import random
import pygame

# ------------------------------------------------------------
# 2D Metropolis sampler visualized with Pygame
# ------------------------------------------------------------
# Idea:
# - State is a point (x, y) in continuous 2D space.
# - Target density is proportional to exp(-beta * U(x, y)).
# - We use a random-walk Metropolis proposal.
# - Pygame visualizes the particle, accepted/rejected moves,
#   and a fading trajectory.
#
# Potential used here:
#     U(x, y) = (x^2 - 1)^2 + 0.35 y^2
# This creates two wells near x = -1 and x = +1.
# ------------------------------------------------------------

WIDTH, HEIGHT = 1000, 720
FPS = 60

# World coordinates shown on screen
X_MIN, X_MAX = -2.6, 2.6
Y_MIN, Y_MAX = -2.0, 2.0

# Metropolis parameters
TEMPERATURE = 0.22
BETA = 1.0 / TEMPERATURE
STEP_SIZE = 0.16
STEPS_PER_FRAME = 8

# Visualization
TRAIL_MAX = 700
POINT_RADIUS = 6
BG = (12, 14, 18)
TEXT = (230, 230, 230)
GRID = (45, 50, 58)
PARTICLE = (80, 220, 255)
ACCEPT_FLASH = (70, 190, 90)
REJECT_FLASH = (210, 80, 80)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Metropolis Sampler")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)
small_font = pygame.font.SysFont("consolas", 16)


def U(x: float, y: float) -> float:
    return (x * x - 1.0) ** 2 + 0.35 * y * y


def proposal(x: float, y: float, step: float) -> tuple[float, float]:
    dx = random.uniform(-step, step)
    dy = random.uniform(-step, step)
    return x + dx, y + dy


def metropolis_step(x: float, y: float, beta: float, step: float) -> tuple[float, float, bool]:
    xn, yn = proposal(x, y, step)
    dU = U(xn, yn) - U(x, y)
    p_acc = min(1.0, math.exp(-beta * dU))
    if random.random() < p_acc:
        return xn, yn, True
    return x, y, False


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    sx = int((x - X_MIN) / (X_MAX - X_MIN) * WIDTH)
    sy = int(HEIGHT - (y - Y_MIN) / (Y_MAX - Y_MIN) * HEIGHT)
    return sx, sy


def screen_to_world(sx: int, sy: int) -> tuple[float, float]:
    x = X_MIN + sx / WIDTH * (X_MAX - X_MIN)
    y = Y_MIN + (HEIGHT - sy) / HEIGHT * (Y_MAX - Y_MIN)
    return x, y


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def draw_grid() -> None:
    for gx in range(-2, 3):
        sx, _ = world_to_screen(gx, 0)
        pygame.draw.line(screen, GRID, (sx, 0), (sx, HEIGHT), 1)
    for gy in [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]:
        _, sy = world_to_screen(0, gy)
        pygame.draw.line(screen, GRID, (0, sy), (WIDTH, sy), 1)

    sx0, _ = world_to_screen(0, 0)
    _, sy0 = world_to_screen(0, 0)
    pygame.draw.line(screen, (80, 86, 98), (sx0, 0), (sx0, HEIGHT), 2)
    pygame.draw.line(screen, (80, 86, 98), (0, sy0), (WIDTH, sy0), 2)


def draw_trail(trail: list[tuple[float, float]]) -> None:
    n = len(trail)
    if n < 2:
        return
    for i, (x, y) in enumerate(trail):
        alpha_like = i / max(1, n - 1)
        r = int(40 + 40 * alpha_like)
        g = int(70 + 130 * alpha_like)
        b = int(120 + 110 * alpha_like)
        sx, sy = world_to_screen(x, y)
        pygame.draw.circle(screen, (r, g, b), (sx, sy), 2)


def draw_potential_hud(x: float, y: float) -> None:
    lines = [
        f"T = {TEMPERATURE:.3f}",
        f"beta = {BETA:.3f}",
        f"step size = {STEP_SIZE:.3f}",
        f"position = ({x:+.3f}, {y:+.3f})",
        f"U(x,y) = {U(x, y):.4f}",
        "controls:",
        "  up/down   : change temperature",
        "  left/right: change step size",
        "  space     : pause/resume",
        "  r         : reset to center",
        "  c         : clear trail",
        "  mouse     : move particle",
    ]
    y0 = 14
    for line in lines:
        surf = small_font.render(line, True, TEXT)
        screen.blit(surf, (14, y0))
        y0 += 20


def draw_status(accept_rate: float, accepted_last: bool, total_steps: int) -> None:
    color = ACCEPT_FLASH if accepted_last else REJECT_FLASH
    label = "accepted" if accepted_last else "rejected"
    text = f"last move: {label}    acceptance rate: {accept_rate:.3f}    steps: {total_steps}"
    surf = font.render(text, True, color)
    rect = surf.get_rect(midbottom=(WIDTH // 2, HEIGHT - 12))
    screen.blit(surf, rect)


x, y = 0.0, 0.0
trail: list[tuple[float, float]] = [(x, y)]
paused = False
accepted_moves = 0
total_moves = 0
accepted_last = True
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                x, y = 0.0, 0.0
                trail = [(x, y)]
                accepted_moves = 0
                total_moves = 0
            elif event.key == pygame.K_c:
                trail = [(x, y)]
            elif event.key == pygame.K_UP:
                TEMPERATURE = clamp(TEMPERATURE + 0.02, 0.05, 2.0)
                BETA = 1.0 / TEMPERATURE
            elif event.key == pygame.K_DOWN:
                TEMPERATURE = clamp(TEMPERATURE - 0.02, 0.05, 2.0)
                BETA = 1.0 / TEMPERATURE
            elif event.key == pygame.K_RIGHT:
                STEP_SIZE = clamp(STEP_SIZE + 0.02, 0.02, 1.0)
            elif event.key == pygame.K_LEFT:
                STEP_SIZE = clamp(STEP_SIZE - 0.02, 0.02, 1.0)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            x, y = screen_to_world(mx, my)
            trail.append((x, y))
            if len(trail) > TRAIL_MAX:
                trail = trail[-TRAIL_MAX:]

    if not paused:
        for _ in range(STEPS_PER_FRAME):
            x, y, accepted_last = metropolis_step(x, y, BETA, STEP_SIZE)
            total_moves += 1
            if accepted_last:
                accepted_moves += 1
            trail.append((x, y))
            if len(trail) > TRAIL_MAX:
                trail = trail[-TRAIL_MAX:]

    accept_rate = accepted_moves / total_moves if total_moves else 0.0

    screen.fill(BG)
    draw_grid()
    draw_trail(trail)

    sx, sy = world_to_screen(x, y)
    pygame.draw.circle(screen, PARTICLE, (sx, sy), POINT_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), (sx, sy), POINT_RADIUS, 1)

    draw_potential_hud(x, y)
    draw_status(accept_rate, accepted_last, total_moves)

    title = font.render("2D double-well Metropolis simulation", True, TEXT)
    screen.blit(title, (WIDTH - title.get_width() - 18, 14))

    wells = [(-1.0, 0.0), (1.0, 0.0)]
    for wx, wy in wells:
        px, py = world_to_screen(wx, wy)
        pygame.draw.circle(screen, (180, 120, 255), (px, py), 9, 1)

    if paused:
        surf = font.render("PAUSED", True, (255, 220, 120))
        screen.blit(surf, (WIDTH - 120, 46))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
