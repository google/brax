import glfw
import rollout

class ViewerState:
    def __init__(self):
        self.cur_eval = 0
        self.cur_env = 0
        self.change_rollout = True
        self.pause = False
        self.show_metrics = False
        self.show_pixel_obs = False
        self.show_help = True

    def key_callback(self, keycode):
        if keycode == glfw.KEY_RIGHT:
            self.change_rollout = True
            self.cur_env += 1
        elif keycode == glfw.KEY_LEFT:
            self.change_rollout = True
            self.cur_env -= 1
        elif keycode == glfw.KEY_UP:
            self.change_rollout = True
            self.cur_eval += 1
        elif keycode == glfw.KEY_DOWN:
            self.change_rollout = True
            self.cur_eval -= 1
        else:
            try:
                char = chr(keycode)
                if char == "M":
                    self.show_metrics = not self.show_metrics
                elif char == "O":
                    self.show_pixel_obs = not self.show_pixel_obs
                elif char == " ":
                    self.pause = not self.pause
                elif char == "H":
                    self.show_help = not self.show_help
            except ValueError:
                pass
        
        # Wrap to valid ranges
        self.cur_eval = (self.cur_eval + rollout.num_evals) % rollout.num_evals
        self.cur_env = (self.cur_env + rollout.num_envs) % rollout.num_envs
