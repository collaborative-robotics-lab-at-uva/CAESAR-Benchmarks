import os
from collections import defaultdict

class TextLogger:
    def __init__(self,
                base_dir, 
                filename,
                print_console=True):
        
        self.base_dir = base_dir
        self.filename = filename
        self.print_console = print_console
    
    def log(self, log_txt, print_console=None):
        if print_console is not None:
            tm_print_console = print_console
        else:
            tm_print_console = self.print_console
        log_file_path = self.base_dir+'/'+self.filename
        os.makedirs(self.base_dir, exist_ok=True)

        with open(log_file_path, 'a') as file:
            file.write(log_txt)

        if tm_print_console:
            print(log_txt)
