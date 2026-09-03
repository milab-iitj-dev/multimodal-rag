# MMRAG Healthcare & Scientific Document Visualizer

A modern, interactive, and responsive web application designed for Multimodal Retrieval-Augmented Generation (RAG) analysis on Chest X-Rays and Scientific Research Literature. 

## Key Features

1. **Dual Domain Toggle (Healthcare | Scientific)**
   * Switch dynamically between Clinical Healthcare and Research Scientific views in the top navigation header.
   * Collapses the PACS radiographic image viewer in Scientific mode to display a clean PDF document viewer for paper analysis.

2. **The 'Evidence & Retrieval' Sliding Drawer**
   * Dedicated sliding panel containing source citations, retrieved snippets, and pipeline metadata.
   * Renders document relevance and confidence scores using float-to-percentage progress bars and high-contrast color badges.

3. **Clinical Memory Graph (Knowledge Graph Layer)**
   * Fully custom interactive node-link visualization container showing disease relations, anatomical structures, and concepts.
   * Implements fluid physics simulations (repulsion, Hooke spring force, gravity) and custom scaled drag-and-drop pointer tracking with zero coordinate lag.

4. **Advanced 'Thinking' Stepper UX**
   * Multi-stage stepper visualization showing step-by-step processing during RAG backend API latency:
     * *Scanning Vector Indices...*
     * *Fusing Modalities...*
     * *Generating Response...*
     * *Verifying Correctness...*

5. **PACS Radiography Toolbar**
   * Fine-tune simulated X-Ray viewports with zoom, brightness, contrast adjustment sliders, and color inversion.
   * Pulse-animated Region-of-Interest (ROI) overlay boundary box highlighting clinical findings when "Highlight Region" is toggled.

## Tech Stack

* **Core:** React 19, TypeScript, Vite, HTML5 SVG Canvas
* **Styling & Animations:** TailwindCSS, Motion (Framer Motion)
* **Icons:** Lucide React
* **AI Integration:** Google Gemini API (`@google/genai` SDK with `gemini-2.5-flash` model)

---

## Run Locally

### Prerequisites
* [Node.js](https://nodejs.org/) installed on your machine.
* A Google Gemini API Key.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/creativekhush17/MMRAG-Healthcare-UI.git
   cd MMRAG-Healthcare-UI
   ```

2. **Install project dependencies:**
   ```bash
   npm install
   ```

3. **Setup environment variables:**
   Create a `.env.local` file in the root directory and add your Gemini API key:
   ```text
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Launch the local development server:**
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser to view the application.
