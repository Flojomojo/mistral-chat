#!/usr/bin/env python

import argparse
import logging
import os
import sys
from rich.console import Console 
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
import pyperclip
import re
import hashlib

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import readline
readline.set_auto_history(True)

MODEL_LIST = [
    "mistral-tiny",
    "mistral-small",
    "mistral-medium",
]

DEFAULT_MODEL = "mistral-small"

COMMANDS = [
    "/help",
    "/quit",
    "/model",
    "/new",
    "/copy",
    "/ccopy"
]

CONSOLE = Console()

# This is kind of weird, but if the status is active and the user CTRL+C the spinner wont stop
# This way we can stop the spinner on CTRL+C
STATUS = Status("Generating answers...", spinner="bouncingBall")

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logger = logging.getLogger("chatbot")

class CodeBlock:
    tag = ""
    raw_code = ""
    code = ""
    identifier = ""
    
    def __init__(self, code: str):
        """
        A class for storing code codeblocks
        code (str): The actual code, including the ```
        """
        self.raw_code = code
        self.tag, self.code = self._extract_code(self.raw_code)
        self.identifier = self._generate_small_hash(self.raw_code)


    def _extract_code(self, input_string):
        """
        Extracts the tag and the code of a given block

        Returns:
            tag(str), code content(str)
        """
        pattern = r'```(\w*)([\s\S]*?)\s*```'
        match = re.search(pattern, input_string)

        if match:
            tag = match.group(1).strip()
            code_content = match.group(2).strip()
            return tag, code_content
        else:
            return None, None
        
    def _generate_small_hash(self, data):
        """Generates a small hash from a string"""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(data.encode('utf-8'))
        small_hash = sha256_hash.hexdigest()[:8]  
        return small_hash

    @staticmethod
    def parse_code_blocks(message: str) -> list:
        """
        Parses the give message to code CodeBlocks

        Args:
            message(str): The message to parse the code blocks from

        Returns:
            A list of parsed code blocks
        """
        pattern = r'```(?:\w+\s*)?\n([\s\S]*?)\s*```'
        matches = re.finditer(pattern, message)

        code_blocks = []
        for match in matches:
            text = match.group()
            code_blocks.append(CodeBlock(text))

        return code_blocks
        


class ChatBot:
    def __init__(self, api_key: str, model: str, streamed: bool, system_message:str=None):
        """
        Initialises the chatbot
        api_key (str): The MistralAi Api Key
        model (str): The name of the model (mistral-tiny, mistral-small, mistral-medium)
        streamed (bool): Wether the output should be streamed
        system_message (bool, Optional): The system message
        """
        self.client = MistralClient(api_key=api_key)
        self.model = model
        self.system_message = system_message
        self.streamed = streamed
        self.first_chat = True
        self.code_blocks = []

    def print_help(self):
        """
        Prints out the help message
        """
        help_message = """
To chat: type your message and hit enter
To start a new chat: type /new
To exit: /quit
To show this help: /help
To switch models: /model {model_name}
Copy the last answer to your clipboard: /copy
Copy a code block via a tag (always above the codeblock): /ccopy <code block tag>
"""
        help_panel = Panel(help_message, subtitle="/help", title="Help", border_style="purple")
        CONSOLE.print(help_panel)

    def new_chat(self):
        """
        Creates a new chat by resetting the seen messages
        """
        CONSOLE.print("Starting new chat...")
        self.messages = []
        readline.clear_history()
        if not self.first_chat:
            CONSOLE.clear()
        self.first_chat = False
        if self.system_message:
            self.messages.append(ChatMessage(role="system", content=self.system_message))


    def check_command(self, content):
        """
        Checks if content is a command and handles it accordingly
        Returns:
            bool, True if content is a command
        """
        normalized_content = content.lower().strip()
        command_parts = normalized_content.split(" ")
        if command_parts[0] in COMMANDS:
            self.handle_command(command_parts[0], command_parts[1::])
            return True
        return False

    def print_available_models(self):
        """
        Prints all the currently available models
        """
        print("Available models are:")
        for model in MODEL_LIST:
            # Print a * to indicate the currently chosen model
            if model == self.model:
                print(f"\t * {model}")
            else:
                print(f"\t - {model}")

    def switch_model(self, args):
        """
        Switches the current model to the new one
        Args:
            args (list): The list of args passed on from handle_command.
                         args[0] should be the name of the model
        """
        if len(args) < 1:
            print("No model provided")
            self.print_available_models()
            return

        model_name = args[0]
        if model_name not in MODEL_LIST:
            print(f"Invalid model name {model_name}")
            self.print_available_models()
            return
        self.model = model_name
        print(f"Successfully switched to model {model_name}")


    def copy_last_message(self):
        """Copies the last message to the clipboard"""
        if len(self.messages) == 0:
            print("No messages yet")
            return
        last_message = self.messages[-1].content
        pyperclip.copy(last_message)

    def copy_code(self, args):
        if len(args) < 1:
            print("No identifier specified")
        identifier = args[0]
        for block in self.code_blocks:
            if block.identifier == identifier:
                pyperclip.copy(block.code)
                print(f"Copied {identifier}")
                return


    def handle_command(self, command, args):
        """
        Handles a command and its args
        Args:
            command (str): The actual command
            args (list): The list of args from the command
        """
        match command:
            case "/new":
                self.new_chat()
            case "/quit":
                self.exit()
            case "/help":
                self.print_help()
            case "/model":
                self.switch_model(args)
            case "/copy":
                self.copy_last_message()
            case "/ccopy":
                self.copy_code(args)

    def inject_code_blocks(self, code_blocks: list[CodeBlock], message: str) -> str:
        # Find the first newline
        new_message = message
        for block in code_blocks:
            raw_code_block_code = block.raw_code
            # Just prepend the identifier so it is on the line before
            new_code = f"\n`{block.identifier}`\n" + raw_code_block_code
            new_message = new_message.replace(raw_code_block_code, new_code)
        return new_message

    def run_inference(self, content):
        """
        Makes the api call to run the model
        Args:
            content (str): The message from the user
        """
        self.messages.append(ChatMessage(role="user", content=content))

        assistant_response = ""
        answer = ""
        logger.debug(f"Sending messages: {self.messages}")
        # If we dont output streamed print a loading message so the user doesnt think nothings happening
        if not args.streamed:
            STATUS.start()
        for chunk in self.client.chat_stream(model=self.model, messages=self.messages):
            response = chunk.choices[0].delta.content
            if response is not None:
                assistant_response += response
                # If streamed output just output with end="" via normal print
                # If not streamed add it to the end string that will be printed out
                if self.streamed:
                    print(response, end="", flush=True)
                answer += response
        
        # We are done so stop the loading
        if not args.streamed:
            STATUS.stop()

        # We havent printed out the result so do it here
        if not self.streamed:
            # Parse the code blocks and inject the code block identifier
            parsed_blocks = CodeBlock.parse_code_blocks(answer)
            self.code_blocks += parsed_blocks
            modified_answer = self.inject_code_blocks(parsed_blocks, answer)
            markdown = Markdown(modified_answer)
            panel = Panel(markdown, title="Mistral", border_style="bold blue")
            CONSOLE.print(panel)
        else:
            print("", flush=True)


        if assistant_response:
            self.messages.append(ChatMessage(role="assistant", content=assistant_response))
        logger.debug(f"Current messages: {self.messages}")

    def get_input(self, prompt: str):
        lines = [] 
        try:
            while True:
                lines.append(input(prompt))
        except EOFError:
            pass
        return "\n".join(lines)


    def start(self):
        self.print_help()
        self.new_chat()
        while True:
            try:
                content = input("> ")
                # Check if a the user input is a command
                if self.check_command(content):
                    continue
                # If streamed output, print a little MISTRAL to emulate a chat
                if self.streamed:
                    print("MISTRAL: ")
                self.run_inference(content)

            # Dont stop on CTRL-C instead stop the current thing and print out a help message
            except KeyboardInterrupt:
                STATUS.stop()
                print("Use /quit to quit")
            except Exception as e:
                print("Error: ", e)


    def exit(self):
        logger.debug("Exiting chatbot")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple chatbot using the Mistral API")
    parser.add_argument("--api-key", default=os.environ.get("MISTRAL_API_KEY"),
                        help="Mistral API key. Defaults to environment variable MISTRAL_API_KEY")
    parser.add_argument("-m", "--model", choices=MODEL_LIST,
                        default=DEFAULT_MODEL,
                        help="Model for chat inference. Choices are %(choices)s. Defaults to %(default)s")
    parser.add_argument("-s", "--system-message",
                        help="Optional system message to prepend.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--streamed", action="store_true", default=False, help="Wether to use streamed output. Disables pretty markdown rendering")

    args = parser.parse_args()

    if args.api_key is None:
        logger.critical("No API key provided or found in environment variables!")
        exit(0)

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Setup logging
    formatter = logging.Formatter(LOG_FORMAT)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.debug(f"Starting chatbot with model: {args.model}")

    bot = ChatBot(args.api_key, args.model, args.streamed, args.system_message)
    bot.start()

