import gradio as gr
from langgraph.graph import StateGraph, END
from state import FloodPredictionState
from config import CONFIG
from logger import structured_log
from data_loader import DataLoaderAgent
from preprocessor import PreprocessorAgent
from model_trainer import ModelTrainerAgent
from model_tuner import ModelTunerAgent
from explainer import ExplainerAgent
from visualizer import VisualizerAgent
from monitor import MonitorAgent
from model_saver import ModelSaverAgent
from predictor import PredictorAgent
from dashboard import DashboardAgent

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
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(FloodPredictionState)
        graph.add_node("load_data", self.data_loader.load_data)
        graph.add_node("preprocess_data", self.preprocessor.preprocess_data)
        graph.add_node("train_models", self.model_trainer.train_models)
        graph.add_node("tune_best_model", self.model_tuner.tune_best_model)
        graph.add_node("explain_model", self.explainer.explain_model)
        graph.add_node("visualize_data", self.visualizer.visualize_data)
        graph.add_node("monitor_performance", self.monitor.monitor_performance)
        graph.add_node("save_model", self.model_saver.save_model)
        graph.add_node("make_sample_prediction", self.predictor.make_sample_prediction)
        graph.add_node("setup_dashboard", self.dashboard.setup_dashboard)
        graph.add_edge("load_data", "preprocess_data")
        graph.add_edge("preprocess_data", "train_models")
        graph.add_edge("train_models", "tune_best_model")
        graph.add_edge("tune_best_model", "explain_model")
        graph.add_edge("explain_model", "visualize_data")
        graph.add_edge("visualize_data", "monitor_performance")
        graph.add_edge("monitor_performance", "save_model")
        graph.add_edge("save_model", "make_sample_prediction")
        graph.add_edge("make_sample_prediction", "setup_dashboard")
        graph.add_edge("setup_dashboard", END)
        graph.set_entry_point("load_data")
        return graph.compile()

    def run(self):
        try:
            structured_log('INFO', "Starting flood prediction pipeline")
            initial_state = FloodPredictionState(data_path=self.config['data_path'])
            final_state = self.graph.invoke(initial_state)
            structured_log('INFO', "Pipeline completed successfully")
            return final_state
        except Exception as e:
            structured_log('ERROR', f"Pipeline failed: {str(e)}")
            raise

# Update config for Hugging Face Spaces
CONFIG['data_path'] = 'data/flood.csv'  # Path relative to Space root
CONFIG['output_dir'] = 'models'  # Output directory in Space

# Run pipeline and get Gradio app
workflow = FloodPredictionWorkflow(CONFIG)
final_state = workflow.run()
app = workflow.dashboard.app  # Gradio app from dashboard.py

# Launch Gradio app (Hugging Face Spaces handles this automatically)
if __name__ == "__main__":
    app.launch()