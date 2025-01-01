import subprocess
def code_runner(code: str,lang: str) -> str:
    if lang == "python":
        # Save the code to a file
        with open("script.py", "w") as f:
            f.write(code)

        # Run the script using subprocess
        result = subprocess.run(["python3", "script.py"], capture_output=True, text=True)
        try:
            subprocess.run(["rm", "script.py"], check=True)
            print("File deleted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
        except FileNotFoundError:
            print("The file does not exist.")
        return result.stdout
