import mlflow

def get_or_create_experiment(experiment_location: str) -> None:
 
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)