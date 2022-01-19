"""
Sample script to start a jupyter lab
"""
import subprocess
import sys

TASK_COMMAND = "jupyter lab --allow-root --no-browser --port=8080 --ip=0.0.0.0"

if __name__ == "__main__":
    print(f"jupyter lab is starting ...")
    args_list = list(TASK_COMMAND.split(' '))
    process = None
    try:
        process = subprocess.Popen(args_list,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=False)
    except FileNotFoundError as e:
        print(f"Is jupyter lab installed?")
        exit(-1)

    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

    if process:
        process.wait()
