class Task:
    def __init__(self, description,inputs,subtasks):
        self.description = description
        self.inputs = inputs
        self.subtasks = subtasks
        self.agentic_sys = None     #list of SubTask objects

    def __str__(self):
        return f'DESCRIPTION: {self.description}\n INPUTS:- {self.inputs}'



# class SubTask:          



# class SubSubTask:
