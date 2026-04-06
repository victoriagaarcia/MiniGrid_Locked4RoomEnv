import gymnasium as gym
import minigrid
import pygame

def main():
    env = gym.make("MiniGrid-LockedRoom-v0", render_mode="human")
    obs, info = env.reset()

    print("Mission:", obs["mission"])
    print("Controls:")
    print("  Left arrow  -> turn left")
    print("  Right arrow -> turn right")
    print("  Up arrow    -> move forward")
    print("  Space       -> pickup")
    print("  Enter       -> toggle/open door")
    print("  Backspace   -> reset environment")
    print("  Esc         -> quit")

    clock = pygame.time.Clock()
    running = True

    while running:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                action = None

                if event.key == pygame.K_LEFT:
                    action = 0   # left
                elif event.key == pygame.K_RIGHT:
                    action = 1   # right
                elif event.key == pygame.K_UP:
                    action = 2   # forward
                elif event.key == pygame.K_SPACE:
                    action = 3   # pickup
                elif event.key == pygame.K_RETURN:
                    action = 5   # toggle
                elif event.key == pygame.K_BACKSPACE:
                    obs, info = env.reset()
                    print("\n--- Reset ---")
                    print("Mission:", obs["mission"])
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)

                    print(
                        f"action={action}, reward={reward:.3f}, "
                        f"terminated={terminated}, truncated={truncated}"
                    )

                    if terminated or truncated:
                        print("\nEpisode finished.")
                        print("Mission:", obs["mission"])
                        obs, info = env.reset()
                        print("\n--- New episode ---")
                        print("Mission:", obs["mission"])

        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()