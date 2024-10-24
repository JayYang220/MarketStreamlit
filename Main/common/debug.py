

def debug_msg(is_debug_mode: bool, *msg: str, end="\n"):
    if is_debug_mode:
        for index, text in enumerate(msg):
            if index == len(msg):
                print(text, end="")
            else:
                print(text, end=" ")
        print(end=end)

