from crewai import Agent, Task, Crew, Process
from logger import structured_log

class FloodPredictionWorkflow:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoaderAgent()
        self.preprocessor = PreprocessorAgent(config)
        self.model_trainer = ModelTrainerAgent(config)
        self.model_tuner = ModelTunerAgent(config)
        self.explainer = ExplainerAgent(config)
        self.visualizer = VisualizerAgent(config)
        self.monitor = MonitorAgent()
        self.model_saver = ModelSaverAgent(config)
        self.predictor = PredictorAgent()
        self.dashboard = DashboardAgent(config)
        self.crews = self._build_crews()

    def _build_crews(self):
        # Define agents
        data_loader_agent = Agent(
            role="Data Loader",
            goal="Load the flood prediction dataset",
            backstory="Expert in data ingestion and validation",
            verbose=True,
            allow_delegation=False
        )
        
        preprocessor_agent = Agent(
            role="Data Preprocessor",
            goal="Preprocess data with feature engineering",
            backstory="Specialist in data cleaning and feature creation",
            verbose=True,
            allow_delegation=False
        )
        
        model_trainer_agent = Agent(
            role="Model Trainer",
            goal="Train multiple machine learning models",
            backstory="Machine learning engineer with expertise in regression models",
            verbose=True,
            allow_delegation=False
        )
        
        model_tuner_agent = Agent(
            role="Model Tuner",
            goal="Tune the best-performing model",
            backstory="Optimization expert for machine learning hyperparameters",
            verbose=True,
            allow_delegation=False
        )
        
        explainer_agent = Agent(
            role="Model Explainer",
            goal="Generate feature importance explanations",
            backstory="Data scientist skilled in model interpretability",
            verbose=True,
            allow_delegation=False
        )
        
        visualizer_agent = Agent(
            role="Data Visualizer",
            goal="Create visualizations for model performance and data insights",
            backstory="Data visualization expert with Plotly and Seaborn experience",
            verbose=True,
            allow_delegation=False
        )
        
        monitor_agent = Agent(
            role="Performance Monitor",
            goal="Monitor model performance metrics",
            backstory="Analyst focused on evaluating machine learning models",
            verbose=True,
            allow_delegation=False
        )
        
        model_saver_agent = Agent(
            role="Model Saver",
            goal="Save the best model and visualizations",
            backstory="Data engineer with expertise in model persistence",
            verbose=True,
            allow_delegation=False
        )
        
        predictor_agent = Agent(
            role="Predictor",
            goal="Make predictions using the trained model",
            backstory="Prediction specialist for real-time inference",
            verbose=True,
            allow_delegation=False
        )
        
        dashboard_agent = Agent(
            role="Dashboard Creator",
            goal="Create an interactive Gradio dashboard",
            backstory="UI/UX developer with Gradio expertise",
            verbose=True,
            allow_delegation=False
        )
        
        # Define tasks
        data_loader_task = Task(
            description="Load the flood prediction dataset from the specified path",
            agent=data_loader_agent,
            expected_output="Dictionary containing the loaded dataset",
            function=self.data_loader.load_data
        )
        
        preprocessor_task = Task(
            description="Preprocess the dataset, including feature engineering: create Monsoon_Drainage, Urban_Climate, LandslideRisk, InadequateInfrastructure; drop TopographyDrainage, Deforestation, DeterioratingInfrastructure, DrainageSystems",
            agent=preprocessor_agent,
            expected_output="Dictionary with preprocessed data and train/test splits",
            function=self.preprocessor.preprocess_data
        )
        
        model_trainer_task = Task(
            description="Train LinearRegression, RandomForest, XGBoost, and LightGBM models",
            agent=model_trainer_agent,
            expected_output="Dictionary with trained models and initial metrics",
            function=self.model_trainer.train_models
        )
        
        model_tuner_task = Task(
            description="Tune the best-performing model using default hyperparameters",
            agent=model_tuner_agent,
            expected_output="Dictionary with the tuned best model",
            function=self.model_tuner.tune_best_model
        )
        
        explainer_task = Task(
            description="Generate feature importance explanations for the best model",
            agent=explainer_agent,
            expected_output="Dictionary with feature importance data",
            function=self.explainer.explain_model
        )
        
        visualizer_task = Task(
            description="Create visualizations for model performance, feature importance, and data distributions",
            agent=visualizer_agent,
            expected_output="Dictionary with saved visualization paths",
            function=self.visualizer.visualize_data
        )
        
        monitor_task = Task(
            description="Monitor and log model performance metrics",
            agent=monitor_agent,
            expected_output="Dictionary with performance metrics",
            function=self.monitor.monitor_performance
        )
        
        model_saver_task = Task(
            description="Save the best model and visualizations to the output directory",
            agent=model_saver_agent,
            expected_output="Dictionary with paths to saved artifacts",
            function=self.model_saver.save_model
        )
        
        predictor_task = Task(
            description="Make a sample prediction using the best model",
            agent=predictor_agent,
            expected_output="Dictionary with sample prediction result",
            function=self.predictor.make_sample_prediction
        )
        
        dashboard_task = Task(
            description="Create and launch a Gradio dashboard with sliders and flood background",
            agent=dashboard_agent,
            expected_output="Dictionary with dashboard status",
            function=self.dashboard.setup_dashboard
        )
        
        # Create crews for sequential and parallel execution
        main_crew = Crew(
            agents=[
                data_loader_agent,
                preprocessor_agent,
                model_trainer_agent,
                model_tuner_agent,
                explainer_agent
            ],
            tasks=[
                data_loader_task,
                preprocessor_task,
                model_trainer_task,
                model_tuner_task,
                explainer_task
            ],
            verbose=2
        )
        
        parallel_crew = Crew(
            agents=[visualizer_agent, monitor_agent],
            tasks=[visualizer_task, monitor_task],
            process=Process.parallel,
            verbose=2
        )
        
        final_crew = Crew(
            agents=[model_saver_agent, predictor_agent, dashboard_agent],
            tasks=[model_saver_task, predictor_task, dashboard_task],
            verbose=2
        )
        
        return [main_crew, parallel_crew, final_crew]

    def run(self):
        try:
            structured_log('INFO', "Starting flood prediction pipeline with CrewAI")
            state = {'data_path': self.config['data_path']}
            for crew in self.crews:
                state = crew.kickoff(inputs=state)
            structured_log('INFO', "Pipeline completed successfully")
            return state
        except Exception as e:
            structured_log('ERROR', f"Pipeline failed: {str(e)}")
            raise