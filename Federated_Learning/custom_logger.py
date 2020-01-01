from __future__ import print_function
from datetime import datetime
import pprint, os

class CustomLogger(object):
    
    LOGGING_LEVEL = {"debug": 4, "step": 3, "info": 2, "none": 0}
    COLOR = {"DEBUG": "\033[94m", "STEP": "\033[92m", "INFO": "\033[93m", "ERROR": "\033[01m\033[91m"}

    def __init__(self, logging_level, logging_class, output_to_file=None):
        self.level = CustomLogger.LOGGING_LEVEL[logging_level]
        self.prettifier = pprint.PrettyPrinter(indent=4, width=120)
        self.class_name = logging_class
        self.file_name = output_to_file
        if self.file_name: # Make log dir if not exists
            os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
      

    def debug(self, msg):
        if self.level >= CustomLogger.LOGGING_LEVEL["debug"]:
            self._msg("DEBUG", msg)

    def step(self, msg):
        if self.level >= CustomLogger.LOGGING_LEVEL["step"]:
            self._msg("STEP", msg)

    def info(self, msg):
        if self.level >= CustomLogger.LOGGING_LEVEL["info"]:
            self._msg("INFO", msg)
            
    def err(self, msg):
        # No Checks because we always want to see errors
        self._msg("ERROR", msg)
        
    def prettify(self, msg):
        return self.prettifier.pformat(msg)

    def _msg(self, level, msg):
        time_formatted = CustomLogger._get_currtime_formatted()
        msg_formatted = "[{}][{:<5} - {}]: {}".format(time_formatted,
                level, self.class_name, msg)
        CustomLogger._output(msg_formatted, self.file_name, color=CustomLogger.COLOR[level])

    @staticmethod
    def _get_currtime_formatted():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def _output(msg_formatted, file_name, color=""):
        if file_name:
            with open(file_name, 'a') as fp:
                fp.write(msg_formatted + "\n")
        else :
            msg_colored =  color + msg_formatted + "\033[00m"
            print(msg_colored, flush=True)
        
    @staticmethod
    def show(logging_subj, msg, output_to_file=None):
        '''
        Use for general outputs
        '''
        time_formatted = CustomLogger._get_currtime_formatted()
        msg_formatted = "[{}][{}]: {}".format(time_formatted,logging_subj, msg)
        CustomLogger._output(msg_formatted, output_to_file)

    @staticmethod
    def log_as_CSV(*args, filename=None):
        '''
        Print without any extra stuff, just with commas between every argument
        '''
        formatted_str = ""
        for arg in args:
            formatted_str += str(arg) + ","
        CustomLogger._output(formatted_str, filename) 
    
    @staticmethod
    def log_as_TSV(*args, filename=None):
        '''
        Print without any extra stuff, just with commas between every argument
        '''
        formatted_str = ""
        for arg in args:
            formatted_str += str(arg) + "\t"
        CustomLogger._output(formatted_str, filename)


def test(output_file):
    logger1 = CustomLogger("debug", "Testing 1", output_file)
    logger1.debug("debug test")
    logger1.step("step test")
    logger1.info("info test")
    logger1.err("err test")

    logger1 = CustomLogger("step", "Testing 2", output_file)
    logger1.debug("debug test")
    logger1.step("step test")
    logger1.info("info test")
    logger1.err("err test")

    logger1 = CustomLogger("info", "Testing 3", output_file)
    logger1.debug("debug test")
    logger1.step("step test")
    logger1.info("info test")
    logger1.err("err test")

    logger1 = CustomLogger("none", "Testing 4", output_file)
    logger1.debug("debug test")
    logger1.step("step test")
    logger1.info("info test")
    logger1.err("err test")

def testCSV(output_file):
    CustomLogger.log_as_CSV("1", 2, [3,4], (5,6), filename=output_file)

if __name__ == "__main__":
    test(None)
    test("temp.log")
    testCSV(None)
    testCSV("temp.log")
