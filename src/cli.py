"""
Command-line interface for EDA-SIM-AI placement engine.
Provides unified access to all major functionalities.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
import yaml
import torch

from .utils.logger import setup_logger
from .training.train_imitation import ImitationTrainer
from .training.train_gnn_surrogate import SurrogateTrainer
from .training.train_rl import PPOTrainer, RLConfig
from .evaluation.benchmark import PlacementBenchmark, CompetitiveBenchmark
from .inference import PlacementInference


def main():
    """Main entry point for EDA-SIM-AI CLI."""
    parser = argparse.ArgumentParser(
        description="EDA-SIM-AI: Learning-Based Chip Placement Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data from Verilog
  eda-sim-ai generate --design designs/aes.v --output data/aes/
  
  # Train imitation learning model
  eda-sim-ai train --phase imitation --config configs/imitation_config.yaml
  
  # Run RL fine-tuning
  eda-sim-ai train --phase rl --config configs/rl_config.yaml --pretrained models/il_model.pth
  
  # Perform placement on new design
  eda-sim-ai place --design new_design.v --model models/best_model.pth --output placements/
  
  # Run benchmarks
  eda-sim-ai benchmark --model models/best_model.pth --suite ispd2005
        """
    )
    
    # Global arguments
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data generation
    gen_parser = subparsers.add_parser('generate', 
                                      help='Generate training data using OpenROAD')
    gen_parser.add_argument('--design', type=str, required=True,
                           help='Input design file (.v or .def)')
    gen_parser.add_argument('--constraints', type=str,
                           help='Timing constraints file (.sdc)')
    gen_parser.add_argument('--output', type=str, required=True,
                           help='Output directory')
    gen_parser.add_argument('--openroad-flow', type=str,
                           default='scripts/openroad_flow.tcl',
                           help='OpenROAD flow script')
    
    # Training
    train_parser = subparsers.add_parser('train',
                                        help='Train placement models')
    train_parser.add_argument('--phase', type=str, required=True,
                             choices=['imitation', 'surrogate', 'rl'],
                             help='Training phase')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Configuration file')
    train_parser.add_argument('--data', type=str,
                             help='Override data directory from config')
    train_parser.add_argument('--pretrained', type=str,
                             help='Pretrained model path (for RL)')
    train_parser.add_argument('--output', type=str,
                             help='Output directory for models')
    train_parser.add_argument('--resume', type=str,
                             help='Resume from checkpoint')
    train_parser.add_argument('--wandb', action='store_true',
                             help='Enable Weights & Biases logging')
    
    # Inference/Placement
    place_parser = subparsers.add_parser('place',
                                        help='Run placement on a design')
    place_parser.add_argument('--design', type=str, required=True,
                             help='Input design file')
    place_parser.add_argument('--model', type=str, required=True,
                             help='Trained model path')
    place_parser.add_argument('--output', type=str, required=True,
                             help='Output directory')
    place_parser.add_argument('--format', type=str, 
                             choices=['def', 'bookshelf', 'json'],
                             default='def',
                             help='Output format')
    place_parser.add_argument('--optimize', action='store_true',
                             help='Run post-placement optimization')
    place_parser.add_argument('--visualize', action='store_true',
                             help='Generate placement visualization')
    
    # Benchmarking
    bench_parser = subparsers.add_parser('benchmark',
                                        help='Run placement benchmarks')
    bench_parser.add_argument('--model', type=str, required=True,
                             help='Model to benchmark')
    bench_parser.add_argument('--suite', type=str, default='ispd2005',
                             help='Benchmark suite')
    bench_parser.add_argument('--baselines', nargs='+',
                             default=['replace', 'ntuplace'],
                             help='Baseline methods to compare')
    bench_parser.add_argument('--output', type=str, default='benchmark_results',
                             help='Output directory')
    bench_parser.add_argument('--competition', action='store_true',
                             help='Run competition-style evaluation')
    
    # Model analysis
    analyze_parser = subparsers.add_parser('analyze',
                                          help='Analyze trained models')
    analyze_parser.add_argument('--model', type=str, required=True,
                               help='Model to analyze')
    analyze_parser.add_argument('--type', type=str,
                               choices=['weights', 'attention', 'features'],
                               default='weights',
                               help='Analysis type')
    analyze_parser.add_argument('--output', type=str, required=True,
                               help='Output directory for analysis')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Execute command
    if args.command == 'generate':
        generate_data(args)
    elif args.command == 'train':
        train_model(args, device)
    elif args.command == 'place':
        run_placement(args, device)
    elif args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'analyze':
        analyze_model(args, device)
    else:
        parser.print_help()
        sys.exit(1)


def generate_data(args):
    """Generate training data using OpenROAD."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating data for {args.design}")
    
    from .data.generate_openroad_data import OpenROADDataGenerator
    
    generator = OpenROADDataGenerator(
        openroad_flow=args.openroad_flow,
        output_dir=args.output
    )
    
    generator.process_design(
        design_file=args.design,
        constraints_file=args.constraints
    )
    
    logger.info(f"Data generated successfully in {args.output}")


def train_model(args, device):
    """Train placement models based on phase."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    # Override config with command-line arguments
    if args.data:
        config['data']['train_dir'] = args.data
    if args.output:
        config['logging']['checkpoint_dir'] = args.output
        
    # Initialize wandb if requested
    if args.wandb:
        import wandb
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=f"{args.phase}_training"
        )
        
    # Train based on phase
    if args.phase == 'imitation':
        logger.info("Starting imitation learning training...")
        trainer = ImitationTrainer(config, device)
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
            
        trainer.train()
        
    elif args.phase == 'surrogate':
        logger.info("Starting PPA surrogate model training...")
        trainer = SurrogateTrainer(config, device)
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
            
        trainer.train()
        
    elif args.phase == 'rl':
        logger.info("Starting RL fine-tuning...")
        
        if not args.pretrained:
            raise ValueError("RL training requires --pretrained model")
            
        # Load pretrained model
        pretrained_model = load_model(args.pretrained, device)
        
        # Create RL trainer
        rl_config = RLConfig(**config['rl'])
        trainer = create_rl_trainer(
            pretrained_model=pretrained_model,
            config=rl_config,
            device=device,
            checkpoint_dir=config['logging']['checkpoint_dir']
        )
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
            
        trainer.train()
        
    logger.info("Training completed successfully!")


def run_placement(args, device):
    """Run placement on a design."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running placement on {args.design}")
    
    # Create inference engine
    inference = PlacementInference(
        model_path=args.model,
        device=device
    )
    
    # Run placement
    placement_result = inference.place(
        design_file=args.design,
        optimize=args.optimize
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'def':
        inference.save_def(placement_result, output_path / f"{Path(args.design).stem}.def")
    elif args.format == 'bookshelf':
        inference.save_bookshelf(placement_result, output_path / f"{Path(args.design).stem}.pl")
    elif args.format == 'json':
        inference.save_json(placement_result, output_path / f"{Path(args.design).stem}.json")
        
    # Generate visualization if requested
    if args.visualize:
        from .visualization import PlacementVisualizer
        visualizer = PlacementVisualizer()
        visualizer.visualize(
            placement_result,
            output_path / f"{Path(args.design).stem}_placement.png"
        )
        
    # Print summary statistics
    logger.info("Placement completed!")
    logger.info(f"  HPWL: {placement_result['metrics']['hpwl']:.2e}")
    logger.info(f"  WNS: {placement_result['metrics']['wns']:.3f} ns")
    logger.info(f"  Power: {placement_result['metrics']['power']:.2f} mW")
    logger.info(f"  Runtime: {placement_result['runtime']:.2f} s")


def run_benchmark(args):
    """Run placement benchmarks."""
    logger = logging.getLogger(__name__)
    
    if args.competition:
        logger.info("Running competition-style benchmarks...")
        benchmark = CompetitiveBenchmark(args.output)
        results = benchmark.generate_competition_results(
            model_path=args.model,
            test_suite=args.suite
        )
        
        # Print summary
        logger.info("\nCompetition Results Summary:")
        for design, result in results.items():
            logger.info(f"  {design}: HPWL={result['hpwl']:.2e}, Valid={result['valid']}")
            
    else:
        logger.info("Running comprehensive benchmarks...")
        benchmark = PlacementBenchmark(
            benchmark_dir=f"data/benchmarks/{args.suite}",
            output_dir=args.output
        )
        
        df = benchmark.run_all_benchmarks(
            model_path=args.model,
            baselines=args.baselines
        )
        
        # Print summary
        logger.info("\nBenchmark Summary:")
        summary = df.groupby('method')[['hpwl', 'runtime']].mean()
        logger.info(summary.to_string())
        
    logger.info(f"\nDetailed results saved to {args.output}/")


def analyze_model(args, device):
    """Analyze trained models."""
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing model: {args.model}")
    
    from .analysis import ModelAnalyzer
    
    analyzer = ModelAnalyzer(args.model, device)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.type == 'weights':
        analyzer.analyze_weights(output_path)
    elif args.type == 'attention':
        analyzer.analyze_attention(output_path)
    elif args.type == 'features':
        analyzer.analyze_features(output_path)
        
    logger.info(f"Analysis results saved to {args.output}")


def load_model(model_path: str, device: torch.device):
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    from .models.gnn_placer import PlacementGNN
    
    model = PlacementGNN(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def create_rl_trainer(pretrained_model, config, device, checkpoint_dir):
    """Create RL trainer with proper environment setup."""
    from .training.train_rl import PPOTrainer, PlacementEnv
    from .models.ppa_estimator import PPAEstimator
    from .data.dataset import load_netlist_graphs
    
    # Load PPA estimator
    ppa_estimator = PPAEstimator().to(device)
    # In practice, would load pretrained weights
    
    # Load training graphs
    graphs = load_netlist_graphs(config.data['train_dir'])
    
    # Create environment
    env = PlacementEnv(graphs, ppa_estimator, config)
    
    # Create policy network
    from .training.train_rl import PPOPolicy
    policy = PPOPolicy(pretrained_model, env.action_space.shape[0]).to(device)
    
    # Create trainer
    trainer = PPOTrainer(env, policy, config, checkpoint_dir)
    
    return trainer


class PlacementInference:
    """Inference engine for chip placement."""
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(model_path)
        
        # Create feature extractor
        from .data.feature_extractor import FeatureExtractor, FeatureConfig
        self.feature_extractor = FeatureExtractor(FeatureConfig())
        
    def _load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Support both regular and JIT models
        if 'model_state_dict' in checkpoint:
            from .models.gnn_placer import PlacementGNN
            model = PlacementGNN(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
        else:
            # JIT model
            model = torch.jit.load(model_path, map_location=self.device)
            
        model.eval()
        return model
        
    def place(self, design_file: str, optimize: bool = False) -> Dict:
        """Run placement on a design."""
        import time
        start_time = time.time()
        
        # Parse design
        from .data.netlist_parser import parse_design
        graph = parse_design(design_file)
        
        # Extract features
        data = self.feature_extractor.extract(graph).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(data)
            placement = outputs['placement'].cpu().numpy()
            
        # Post-processing
        if optimize:
            placement = self._optimize_placement(placement, graph)
            
        # Calculate final metrics
        from .evaluation.metrics import PlacementMetrics
        metrics_calc = PlacementMetrics()
        metrics = {
            'hpwl': metrics_calc.calculate_hpwl(graph, placement),
            'wns': metrics_calc.calculate_wns(graph, placement),
            'tns': metrics_calc.calculate_tns(graph, placement),
            'power': metrics_calc.estimate_power(graph, placement),
            'congestion': metrics_calc.calculate_congestion(graph, placement)
        }
        
        runtime = time.time() - start_time
        
        return {
            'design': graph,
            'placement': placement,
            'metrics': metrics,
            'runtime': runtime
        }
        
    def _optimize_placement(self, placement, graph):
        """Post-placement optimization."""
        # Simple local search optimization
        # In practice, would use more sophisticated methods
        return placement
        
    def save_def(self, result: Dict, output_path: str):
        """Save placement in DEF format."""
        # Implementation for DEF format
        pass
        
    def save_bookshelf(self, result: Dict, output_path: str):
        """Save placement in Bookshelf format."""
        # Implementation for Bookshelf format
        pass
        
    def save_json(self, result: Dict, output_path: str):
        """Save placement in JSON format."""
        import json
        
        output = {
            'design_name': result['design'].design_name,
            'num_cells': len(result['design'].cells),
            'placement': result['placement'].tolist(),
            'metrics': result['metrics'],
            'runtime': result['runtime']
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()