# Management and Analysis of Physics Dataset - Part B (Data Processing)

## Overview
Real-time streaming data processing project for cosmic ray detection using Drift Tube detectors. The system implements a data pipeline with Kafka for stream processing and real-time dashboard visualization.

## Team Members
- Filippo Conforto (2021856)
- Lorenzo Domenichetti (2011653)
- Tommaso Faorlin (2021857)

## Contents

### Main Components
- **Dashboard.ipynb**: Real-time monitoring dashboard
  - Kafka consumer implementation
  - Live data visualization
  - Cosmic ray event display
  
- **Computation.ipynb**: Data processing and analysis algorithms

- **Producer.py**: Kafka data producer for streaming simulation
- **Producer10k.py**: High-volume data producer

- **Slides.ipynb**: Project presentation notebook
- **Slideshow.html**: Exported presentation

### Supporting Directories
- **Pictures/**: Visual assets and plots
- **Scripts/**: Utility scripts

## Technologies
- **Language**: Python
- **Streaming**: Apache Kafka
- **Key Libraries**: 
  - kafka-python: Stream processing
  - NumPy, Matplotlib: Data analysis and visualization
  - IPython: Interactive dashboard updates
- **Format**: Jupyter Notebooks

## Architecture
1. Producer generates/streams cosmic ray detector data
2. Kafka handles message queuing and distribution
3. Consumer processes data in real-time
4. Dashboard displays live results and statistics
