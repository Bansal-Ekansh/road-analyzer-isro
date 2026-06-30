# Route Resilience Analyzer
**ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 4**

This project provides an AI-powered road network extraction, bottleneck detection, and disaster resilience simulation platform. It uses DeepLabV3+ with a ResNet50 encoder to perform road segmentation from satellite imagery, extracting the road network using OSMnx and analyzing network topology and bottlenecks using NetworkX.

## Features
- **Deep Learning Extraction:** Leverages a custom-trained PyTorch model for road segmentation from RGB satellite or aerial tiles.
- **Graph Modeling:** Converts binary masks into functional road network graphs to identify critical infrastructure nodes.
- **Resilience Simulation:** Automatically calculates "Cascading Failure Index" and simulates failure conditions, finding optimal rerouting strategies dynamically.
- **Interactive Map:** Displays geo-referenced insights, highlighting active routes vs affected nodes.

## Running the Application
To launch the Streamlit dashboard, run the following command from the project root:

```bash
python -m streamlit run app/main.py
```

*Ensure your dependencies from `requirements.txt` are installed and your trained weights `.pth` are placed within `models/`.*
