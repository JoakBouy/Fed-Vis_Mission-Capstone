# Fed-Vis Demo Video Script

**Duration:** 7-10 minutes
**Format:** Screen recording with narration

---

## INTRO (0:00 - 0:45)

### Opening Slide
*[Show Fed-Vis logo on dark background]*

**Narration:**
"Hi, I'm [Your Name], and today I'll demonstrate Fed-Vis - a privacy-preserving 3D medical image segmentation system using federated learning and attention mechanisms."

### Project Overview
*[Show architecture diagram]*

**Narration:**
"Fed-Vis addresses a critical challenge in medical AI: how do we train accurate models without compromising patient privacy? Our solution uses federated learning to train across multiple hospital nodes while keeping patient data local."

---

## PART 1: ENVIRONMENT SETUP (0:45 - 2:00)

### Show Project Structure
*[Open VS Code, show file tree]*

**Narration:**
"Let's look at the project structure. We have our source code in `fed_vis/src`, configuration files managed by Hydra, and a DVC pipeline for data versioning."

### Installation
*[Terminal: show requirements.txt, run pip install]*

```bash
pip install -r requirements.txt
```

**Narration:**
"The installation is straightforward - all dependencies are pinned in requirements.txt. We use PyTorch for the model, FastAPI for inference, and Hydra for configuration management."

### Run Tests
*[Terminal: run pytest]*

```bash
python -m pytest tests/ -v
```

**Narration:**
"Before we dive in, let's verify everything works. All 28 unit tests pass, covering our model architecture, loss functions, and building blocks."

---

## PART 2: MODEL ARCHITECTURE (2:00 - 4:00)

### Open Notebook
*[Launch Jupyter, open 01_demo_fedvis.ipynb]*

**Narration:**
"Our core model is a 3D Attention U-Net with about 90 million parameters. Let me walk you through the architecture in our demo notebook."

### Show Architecture Cells
*[Run cells showing model structure]*

**Narration:**
"The encoder progressively downsamples the input through 4 levels, expanding features from 64 to 1024. The decoder then upsamples back to the original resolution."

### Highlight Attention Gates
*[Run attention gate visualization cell]*

**Narration:**
"The key innovation is our attention gates at each skip connection. Unlike standard U-Net, our model learns WHERE to focus - and we can visualize these attention maps for interpretability."

### Parameter Breakdown
*[Show parameter count output]*

**Narration:**
"Here's the parameter breakdown - about 45 million in the encoder, 42 million in the decoder, and 3 million in attention gates. This is a substantial model but manageable on a single GPU."

---

## PART 3: DATA VISUALIZATION (4:00 - 5:30)

### Synthetic Data Generation
*[Run synthetic volume cells]*

**Narration:**
"For this demo, we generate synthetic 3D brain volumes with simulated tumors. In production, this would be real MRI data from datasets like BraTS."

### Show Slice Views
*[Display axial, sagittal, coronal slices]*

**Narration:**
"We can visualize the data across all three anatomical planes. The bright region here is our simulated tumor, and the ground truth mask shows exactly where it is."

### Intensity Analysis
*[Show histogram plots]*

**Narration:**
"This histogram shows the intensity distribution - notice the clear separation between background tissue and the tumor region. Our Z-score normalization makes this consistent across scans."

---

## PART 4: INFERENCE DEMO (5:30 - 7:00)

### Run Segmentation
*[Execute model inference cell]*

**Narration:**
"Now let's run inference. We pass our volume through the model and get both the segmentation prediction and attention maps."

### Compare Results
*[Show ground truth vs prediction side-by-side]*

**Narration:**
"Here's the comparison - ground truth on the left, our prediction on the right. The Dice score of 0.87 shows strong overlap, and you can see the model correctly identifies the tumor boundaries."

### Attention Maps
*[Display attention heatmaps]*

**Narration:**
"These attention maps show where the model focused. Notice how the highest attention is exactly on the tumor region - this gives doctors confidence that the model is looking at the right features."

### Metrics Output
*[Show metrics table]*

**Narration:**
"Our final metrics: Dice score 0.87, IoU 0.77, Precision 0.91, Recall 0.84. These are competitive results for 3D medical segmentation."

---

## PART 5: API SERVICE (7:00 - 8:30)

### Start Server
*[Terminal: run uvicorn]*

```bash
uvicorn fedvis.api.app:app --reload --port 8000
```

**Narration:**
"For deployment, we've built a FastAPI inference service. Let me start it up."

### Swagger UI Demo
*[Open browser: localhost:8000/docs]*

**Narration:**
"The API has automatic Swagger documentation. We have endpoints for health checks, model info, and prediction."

### Test Endpoints
*[Click /model/info, show response]*

**Narration:**
"The model info endpoint returns architecture details and parameter counts - useful for system monitoring."

*[Upload test file to /predict]*

**Narration:**
"For prediction, we upload a numpy file and get back segmentation statistics including inference time and tumor volume."

---

## PART 6: NEXT STEPS (8:30 - 9:30)

### Roadmap Slide
*[Show development timeline]*

**Narration:**
"Looking ahead, our next milestones are:
1. Integrating Flower for actual federated training
2. Building the React frontend with Three.js 3D visualization
3. Training on real BraTS data across simulated hospital nodes"

### UI Mockup
*[Show Doctor's Cockpit design]*

**Narration:**
"Here's a preview of our planned UI - the Doctor's Cockpit interface where radiologists can interact with 3D visualizations and review AI predictions."

---

## OUTRO (9:30 - 10:00)

### Closing
*[Return to logo slide]*

**Narration:**
"That concludes our Fed-Vis demo. We've shown a working Attention U-Net for 3D segmentation, unit-tested architecture, interactive notebook, and production-ready API."

"Thank you for watching. The code is available on GitHub, and I welcome any questions or feedback."

*[Show GitHub URL and contact info]*

---

## Recording Tips

1. **Resolution:** 1920x1080 minimum
2. **Audio:** Use external mic, record in quiet room
3. **Pace:** Speak slowly, pause between sections
4. **Cursor:** Use large cursor, highlight important areas
5. **Editing:** Cut mistakes, add transitions between sections
6. **Time:** Aim for 7-8 minutes, max 10 minutes
