import time
import threading
import sys
import os


class ModelPrinter:
    def __init__(self, max_line_length=75):
        self.max_line_length = max_line_length

    def print(self, text):
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= self.max_line_length:
                current_line += word + " "
            else:
                print(current_line)
                current_line = word + " "
        print(current_line)


class ExecutionTimer:
    def __init__(self):
        self.start_time = time.time()
        self.current_time = 0
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.update_timer).start()

    def stop(self):
        self.running = False
        self.current_time = time.time() - self.start_time
        sys.stdout.write(f"\rTotal execution time: {self.current_time:.2f} seconds\n")
        sys.stdout.flush()

    def update_timer(self):
        while self.running:
            elapsed = time.time() - self.start_time
            sys.stdout.write(f" ")
            sys.stdout.flush()
            time.sleep(1)
